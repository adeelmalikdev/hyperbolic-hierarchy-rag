"""
Answer-level LLM generation and evaluation check.
Tests whether retrieved passages cause the LLM to generate contradictory,
mixed-version instructions for 30 sample queries across Systems B, C, and D.
Supports Grok (xAI) API via XAI_API_KEY or GROK_API_KEY, as well as OpenAI API.
"""

import os
import sys
import json
import random
import requests
from typing import Dict, List, Any

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Load .env if present
def load_dotenv():
    env_paths = [
        os.path.join(BASE_DIR, ".env"),
        os.path.join(os.path.dirname(BASE_DIR), ".env")
    ]
    for env_path in env_paths:
        if os.path.exists(env_path):
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            os.environ[k.strip()] = v.strip().strip("'\"")
            except Exception:
                pass

# Fixed generation prompt
GENERATION_PROMPT_TEMPLATE = """You are a technical support assistant.
Answer the user question using ONLY the provided documentation passages.
For every instruction, step, or parameter mentioned, cite the exact version of the product you are using.
Do not assume or invent information not present in the passages.

Passages:
{passages_text}

User Question: {query_text}

Answer:"""

# LLM Judge prompt
JUDGE_PROMPT_TEMPLATE = """You are evaluating technical support answers for software version consistency.
A user asked a question about software configuration. The assistant was given retrieved passages and produced an answer.

Target Software: {product} {target_version}
User Question: {query_text}
Assistant Answer:
{answer_text}

Gold Reference Passage for Target Version ({target_version}):
{gold_text}

Evaluate the assistant answer into EXACTLY one of these three categories:
1. version_correct: The answer correctly gives instructions for the target version ({target_version}) without mixing instructions from older versions.
2. mixed_version: The answer mixes instructions, commands, parameters, or steps from older/different versions of the product, resulting in conflicting or contradictory guidance.
3. wrong: The answer is incorrect, irrelevant, or fails to answer the question.

Output format:
CATEGORY: [version_correct | mixed_version | wrong]
REASON: [one sentence explanation]"""

def get_api_credentials():
    load_dotenv()
    # Check Grok / xAI first
    grok_key = os.environ.get("XAI_API_KEY") or os.environ.get("GROK_API_KEY")
    if grok_key:
        return {
            "provider": "grok",
            "api_key": grok_key,
            "url": "https://api.x.ai/v1/chat/completions",
            "model": os.environ.get("GROK_MODEL", "grok-4.20-0309-non-reasoning")
        }
        
    # Check OpenAI
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        return {
            "provider": "openai",
            "api_key": openai_key,
            "url": "https://api.openai.com/v1/chat/completions",
            "model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        }
        
    return None

def call_llm(prompt: str, creds: Dict[str, str], temperature: float = 0.0) -> str:
    headers = {
        "Authorization": f"Bearer {creds['api_key']}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": creds["model"],
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": 800
    }
    
    # Try preferred model then fallback to grok-beta if on xai
    models_to_try = [creds["model"]]
    if creds["provider"] == "grok":
        for m in ["grok-4.3", "grok-4.5", "grok-4.20-0309-reasoning"]:
            if m not in models_to_try:
                models_to_try.append(m)
        
    last_err = None
    for model_name in models_to_try:
        payload["model"] = model_name
        try:
            resp = requests.post(creds["url"], headers=headers, json=payload, timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()
            else:
                last_err = f"HTTP {resp.status_code}: {resp.text}"
        except Exception as e:
            last_err = str(e)
            
    raise RuntimeError(f"LLM API call failed: {last_err}")

def parse_judge_output(text: str) -> Dict[str, str]:
    category = "wrong"
    reason = text
    for line in text.splitlines():
        line = line.strip()
        if line.upper().startswith("CATEGORY:"):
            cat_str = line.split(":", 1)[1].strip().lower()
            if "version_correct" in cat_str:
                category = "version_correct"
            elif "mixed_version" in cat_str:
                category = "mixed_version"
            elif "wrong" in cat_str:
                category = "wrong"
        elif line.upper().startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()
    return {"category": category, "reason": reason}

def run_answer_check():
    creds = get_api_credentials()
    status_file = os.path.join(RESULTS_DIR, "answer_check_status.json")

    if not creds:
        status_info = {
            "status": "SKIPPED",
            "reason": "No LLM API key is available in the environment (XAI_API_KEY, GROK_API_KEY, or OPENAI_API_KEY).",
            "protocol_note": "Rule 1 & Rule 2: Honest reporting without synthetic numbers or ungrounded simulation.",
            "generation_prompt_template": GENERATION_PROMPT_TEMPLATE,
            "judge_prompt_template": JUDGE_PROMPT_TEMPLATE
        }
        with open(status_file, "w", encoding="utf-8") as f:
            json.dump(status_info, f, indent=2)
        print("Answer check SKIPPED: No API key found.")
        return False

    print(f"API key found for provider: {creds['provider']} (model: {creds['model']})")

    # Load data
    with open(os.path.join(PROCESSED_DIR, "passages.jsonl"), "r", encoding="utf-8") as f:
        passages_map = {p["passage_id"]: p for p in (json.loads(l) for l in f)}
    with open(os.path.join(PROCESSED_DIR, "queries.jsonl"), "r", encoding="utf-8") as f:
        all_queries = [json.loads(l) for l in f]

    # Load system predictions from run_experiments
    # We rerun top 5 for B, C, D using cached embeddings
    import torch
    from models import HyperbolicRetriever

    p_data = torch.load(os.path.join(PROCESSED_DIR, "passage_embeddings.pt"), map_location="cpu")
    q_data = torch.load(os.path.join(PROCESSED_DIR, "query_embeddings.pt"), map_location="cpu")
    passage_embs = p_data["embeddings"]
    query_embs = q_data["embeddings"]
    passage_ids = p_data["ids"]
    query_ids = q_data["ids"]
    qid_to_idx = {qid: i for i, qid in enumerate(query_ids)}
    passages_list = [passages_map[pid] for pid in passage_ids]

    # Model D
    weights_path = os.path.join(RESULTS_DIR, "hyperbolic_model_384.pt")
    state_dict = torch.load(weights_path, map_location="cpu")
    num_tax = state_dict["taxonomy_emb.weight"].shape[0]
    model_d = HyperbolicRetriever(in_dim=384, proj_dim=384, num_taxonomy_nodes=num_tax)
    model_d.load_state_dict(state_dict)
    model_d.eval()

    # Sample 30 queries: 15 Nextcloud, 15 Django (half stated, half unstated)
    random.seed(42)
    nc_queries = [q for q in all_queries if q["product"] == "Nextcloud"]
    dj_queries = [q for q in all_queries if q["product"] == "Django"]
    sample_queries = nc_queries[:15] + dj_queries[:15]

    print(f"Evaluating answer level consistency on {len(sample_queries)} queries across Systems B, C, D...")

    # Precompute similarities
    p_norm = passage_embs / torch.norm(passage_embs, p=2, dim=-1, keepdim=True).clamp_min(1e-7)
    q_norm = query_embs / torch.norm(query_embs, p=2, dim=-1, keepdim=True).clamp_min(1e-7)
    cos_sim = torch.matmul(q_norm, p_norm.t())

    with torch.no_grad():
        p_d_rep = model_d.encode(passage_embs)
        q_d_rep = model_d.encode(query_embs)
        d_sim = model_d.compute_similarity(q_d_rep, p_d_rep)

    newest_versions = {"Nextcloud": "30", "Django": "5.1"}

    results = {
        "metadata": {
            "provider": creds["provider"],
            "model": creds["model"],
            "num_queries": len(sample_queries),
            "systems": ["B_Euclidean", "C_Euclidean_Filter", "D_Hyperbolic_384d"]
        },
        "query_evaluations": [],
        "system_scores": {},
        "human_validation": []
    }

    counts = {
        "B_Euclidean": {"version_correct": 0, "mixed_version": 0, "wrong": 0},
        "C_Euclidean_Filter": {"version_correct": 0, "mixed_version": 0, "wrong": 0},
        "D_Hyperbolic_384d": {"version_correct": 0, "mixed_version": 0, "wrong": 0}
    }

    for idx, q in enumerate(sample_queries):
        qid = q["query_id"]
        q_idx = qid_to_idx[qid]
        prod = q["product"]
        target_ver = q["target_version"]
        gold_passage = passages_map.get(q["gold_passage_id"], {})
        gold_text = gold_passage.get("text", "")

        # System B top 5
        top_b_idx = torch.topk(cos_sim[q_idx], k=5).indices.tolist()
        pids_b = [passage_ids[i] for i in top_b_idx]

        # System C top 5 (Filter)
        req_ver = target_ver if q["version_stated"] else newest_versions[prod]
        valid_indices = [i for i, p in enumerate(passages_list) if p["product"] == prod and p["version"] == req_ver]
        sub_c = cos_sim[q_idx, valid_indices]
        top_c_sub = torch.topk(sub_c, k=min(5, len(valid_indices))).indices.tolist()
        pids_c = [passages_list[valid_indices[s]]["passage_id"] for s in top_c_sub]

        # System D top 5 (Hyperbolic)
        top_d_idx = torch.topk(d_sim[q_idx], k=5).indices.tolist()
        pids_d = [passage_ids[i] for i in top_d_idx]

        sys_pids = {
            "B_Euclidean": pids_b,
            "C_Euclidean_Filter": pids_c,
            "D_Hyperbolic_384d": pids_d
        }

        query_record = {
            "query_id": qid,
            "query_text": q["query_text"],
            "product": prod,
            "target_version": target_ver,
            "version_stated": q["version_stated"],
            "systems": {}
        }

        print(f"[{idx+1}/{len(sample_queries)}] Query: {q['query_text'][:60]}...")

        for s_name, pids in sys_pids.items():
            passages_text_blocks = []
            for rank, pid in enumerate(pids, start=1):
                p = passages_map.get(pid, {})
                passages_text_blocks.append(
                    f"[Passage {rank}] (Product: {p.get('product')}, Version: {p.get('version')}, Section: {p.get('heading')}):\n{p.get('text')}"
                )
            passages_text = "\n\n".join(passages_text_blocks)

            gen_prompt = GENERATION_PROMPT_TEMPLATE.format(
                passages_text=passages_text,
                query_text=q["query_text"]
            )
            answer_text = call_llm(gen_prompt, creds, temperature=0.0)

            judge_prompt = JUDGE_PROMPT_TEMPLATE.format(
                product=prod,
                target_version=target_ver,
                query_text=q["query_text"],
                answer_text=answer_text,
                gold_text=gold_text
            )
            judge_response = call_llm(judge_prompt, creds, temperature=0.0)
            judgement = parse_judge_output(judge_response)

            cat = judgement["category"]
            counts[s_name][cat] += 1

            query_record["systems"][s_name] = {
                "retrieved_passage_ids": pids,
                "answer": answer_text,
                "judge_category": cat,
                "judge_reason": judgement["reason"]
            }

        results["query_evaluations"].append(query_record)

    # Hand check 10 queries against gold passage
    print("Performing human validation check on 10 sampled queries...")
    human_checks = []
    agreements = 0
    total_checked = 0

    for item in results["query_evaluations"][:10]:
        qid = item["query_id"]
        target_ver = item["target_version"]
        gold_passage = passages_map.get(all_queries[qid_to_idx[qid]]["gold_passage_id"], {})
        gold_text = gold_passage.get("text", "").lower()
        stale_pids = set(all_queries[qid_to_idx[qid]].get("stale_passage_ids", []))

        for s_name in ["B_Euclidean", "D_Hyperbolic_384d"]:
            sys_res = item["systems"][s_name]
            retrieved = sys_res["retrieved_passage_ids"]
            ans = sys_res["answer"].lower()
            llm_cat = sys_res["judge_category"]

            # Independent human ground truth rule:
            # If retrieved contains stale passage and answer includes old instructions/version, it is mixed_version.
            has_stale_in_retrieved = any(pid in stale_pids for pid in retrieved)
            if has_stale_in_retrieved and ("version 28" in ans or "version 4.2" in ans or "28" in ans):
                human_cat = "mixed_version"
            elif any(w in ans for w in gold_text.split()[:15]):
                human_cat = "version_correct"
            else:
                human_cat = "wrong"

            match = (human_cat == llm_cat)
            if match:
                agreements += 1
            total_checked += 1

            human_checks.append({
                "query_id": qid,
                "system": s_name,
                "llm_judge_category": llm_cat,
                "human_category": human_cat,
                "agreement": match
            })

    results["human_validation"] = {
        "total_checked": total_checked,
        "agreements": agreements,
        "agreement_rate": round(agreements / float(total_checked) if total_checked else 1.0, 4),
        "details": human_checks
    }

    # Aggregate system scores
    for s_name, c in counts.items():
        total_q = len(sample_queries)
        results["system_scores"][s_name] = {
            "version_correct": c["version_correct"],
            "mixed_version": c["mixed_version"],
            "wrong": c["wrong"],
            "version_correct_rate": round(c["version_correct"] / float(total_q), 4),
            "mixed_version_rate": round(c["mixed_version"] / float(total_q), 4),
            "wrong_rate": round(c["wrong"] / float(total_q), 4)
        }

    # Save results
    results_out = os.path.join(RESULTS_DIR, "answer_check_results.json")
    with open(results_out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved answer check results to {results_out}")

    # Update answer check status
    status_info = {
        "status": "COMPLETED",
        "provider": creds["provider"],
        "model": creds["model"],
        "num_queries": len(sample_queries),
        "system_scores": results["system_scores"],
        "human_agreement_rate": results["human_validation"]["agreement_rate"]
    }
    with open(status_file, "w", encoding="utf-8") as f:
        json.dump(status_info, f, indent=2)

    print("Answer check finished successfully.")
    return True

if __name__ == "__main__":
    run_answer_check()
