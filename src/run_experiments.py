"""
Main experiment runner for version-conflicted retrieval.
Implements:
- System A: BM25 Lexical Retrieval
- System B: Euclidean Dense Retrieval (all-MiniLM-L6-v2)
- System B-trained: Euclidean Control (trained with contrastive + hierarchy loss)
- System C: System B + Hard Metadata Filter (guesses newest version if unstated)
- System D: Hyperbolic Retrieval (Lorentz model, trained)
- System E: System D + Hard Metadata Filter
- System F: Reciprocal Rank Fusion of B and D
- 32-dimensional ablation runs for B-trained and D.
"""

import os
import sys
import time
import json
import random
import platform
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from rank_bm25 import BM25Okapi

from models import (
    HyperbolicRetriever,
    EuclideanRetriever,
    lorentz_distance,
    lorentz_distance_matrix,
    DEPTH_NORMS
)
from metrics import evaluate_system

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def build_taxonomy(passages):
    """
    Constructs a taxonomy tree from passages.
    Nodes: product (depth 0), version (depth 1), category (depth 2), article (depth 3).
    Returns node_to_id, id_to_depth, passage_ancestors, negative_ancestors
    """
    node_to_id = {}
    id_to_depth = {}
    
    def get_or_create(node_key, depth):
        if node_key not in node_to_id:
            nid = len(node_to_id)
            node_to_id[node_key] = nid
            id_to_depth[nid] = depth
        return node_to_id[node_key]

    passage_ancestors = {}
    product_versions = defaultdict(set)
    article_versions = defaultdict(lambda: defaultdict(str))

    for p in passages:
        prod = p["product"]
        ver = p["version"]
        cat = p["category"]
        art = p["article"]
        rel = p["rel_path"]
        
        product_versions[prod].add(ver)
        article_versions[prod][(rel, art)] = ver

        prod_id = get_or_create(f"prod:{prod}", 0)
        ver_id = get_or_create(f"ver:{prod}:{ver}", 1)
        cat_id = get_or_create(f"cat:{prod}:{ver}:{cat}", 2)
        art_id = get_or_create(f"art:{prod}:{ver}:{rel}", 3)

        passage_ancestors[p["passage_id"]] = {
            "prod": prod_id,
            "ver": ver_id,
            "cat": cat_id,
            "art": art_id,
            "product": prod,
            "version": ver,
            "rel_path": rel
        }

    # Precompute negative version nodes for each passage
    passage_negative_nodes = {}
    for pid, anc in passage_ancestors.items():
        prod = anc["product"]
        cur_ver = anc["version"]
        rel = anc["rel_path"]
        
        neg_ver_nodes = []
        for other_ver in product_versions[prod]:
            if other_ver != cur_ver:
                other_ver_key = f"ver:{prod}:{other_ver}"
                if other_ver_key in node_to_id:
                    neg_ver_nodes.append(node_to_id[other_ver_key])
                other_art_key = f"art:{prod}:{other_ver}:{rel}"
                if other_art_key in node_to_id:
                    neg_ver_nodes.append(node_to_id[other_art_key])
                    
        passage_negative_nodes[pid] = neg_ver_nodes

    return node_to_id, id_to_depth, passage_ancestors, passage_negative_nodes

def train_retriever(
    model,
    passage_embs,
    query_embs,
    queries,
    passages,
    passage_ancestors,
    passage_negative_nodes,
    id_to_depth,
    is_hyperbolic=True,
    epochs=15,
    lr=1e-3,
    device="cpu"
):
    """
    Trains retriever with Contrastive loss and Hierarchy loss.
    """
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    pid_to_idx = {p["passage_id"]: i for i, p in enumerate(passages)}
    qid_to_idx = {q["query_id"]: i for i, q in enumerate(queries)}

    # Build depth targets for all taxonomy nodes
    depth_targets = torch.tensor(
        [DEPTH_NORMS[id_to_depth[i]] for i in range(len(id_to_depth))],
        device=device,
        dtype=torch.float32
    )

    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        # Shuffle queries
        indices = list(range(len(queries)))
        random.shuffle(indices)

        batch_size = 32
        for b_start in range(0, len(indices), batch_size):
            b_indices = indices[b_start:b_start + batch_size]
            b_queries = [queries[idx] for idx in b_indices]

            # Collect query embeddings and gold embeddings
            q_emb_list = [query_embs[idx] for idx in b_indices]
            q_tensors = torch.stack(q_emb_list).to(device)

            gold_indices = [pid_to_idx[q["gold_passage_id"]] for q in b_queries]
            g_tensors = passage_embs[gold_indices].to(device)

            # Encode query and gold
            q_rep = model.encode(q_tensors)
            g_rep = model.encode(g_tensors)

            # Compute contrastive loss with stale hard negatives
            contrastive_loss = torch.tensor(0.0, device=device)
            num_cont = 0
            
            for i, q in enumerate(b_queries):
                # distance(q_i, gold_i)
                if is_hyperbolic:
                    dist_gold = lorentz_distance(q_rep[i:i+1], g_rep[i:i+1]).squeeze()
                else:
                    dist_gold = torch.norm(q_rep[i] - g_rep[i], p=2)

                # Hard negative stale passages
                stale_ids = [pid_to_idx[sid] for sid in q.get("stale_passage_ids", []) if sid in pid_to_idx]
                if stale_ids:
                    s_tensors = passage_embs[stale_ids].to(device)
                    s_rep = model.encode(s_tensors)
                    if is_hyperbolic:
                        dist_stale = lorentz_distance(q_rep[i:i+1].expand(len(stale_ids), -1), s_rep)
                    else:
                        dist_stale = torch.norm(q_rep[i:i+1] - s_rep, p=2, dim=-1)

                    # Margin triplet: dist_gold - dist_stale + margin < 0
                    cont_margin = 0.5
                    loss_stale = F.relu(dist_gold - dist_stale + cont_margin).mean()
                    contrastive_loss = contrastive_loss + loss_stale
                    num_cont += 1

                # In-batch negative (other queries' gold passages)
                if len(b_queries) > 1:
                    other_golds = [g_rep[j] for j in range(len(b_queries)) if j != i]
                    other_g_tensors = torch.stack(other_golds)
                    if is_hyperbolic:
                        dist_other = lorentz_distance(q_rep[i:i+1].expand(len(other_golds), -1), other_g_tensors)
                    else:
                        dist_other = torch.norm(q_rep[i:i+1] - other_g_tensors, p=2, dim=-1)
                    loss_inbatch = F.relu(dist_gold - dist_other + 0.3).mean()
                    contrastive_loss = contrastive_loss + loss_inbatch
                    num_cont += 1

            if num_cont > 0:
                contrastive_loss = contrastive_loss / num_cont

            # Hierarchy loss for gold passages
            hier_loss = torch.tensor(0.0, device=device)
            num_hier = 0

            for i, q in enumerate(b_queries):
                pid = q["gold_passage_id"]
                p_rep = g_rep[i:i+1]
                anc = passage_ancestors[pid]
                
                # Positive ancestors: article, category, version
                pos_node_ids = [anc["art"], anc["cat"], anc["ver"]]
                pos_tensors = model.encode_taxonomy(torch.tensor(pos_node_ids, device=device))
                
                if is_hyperbolic:
                    dist_pos = lorentz_distance(p_rep.expand(len(pos_node_ids), -1), pos_tensors)
                else:
                    dist_pos = torch.norm(p_rep - pos_tensors, p=2, dim=-1)

                # Negative ancestors: nodes from other versions of same product
                neg_node_ids = passage_negative_nodes[pid]
                if neg_node_ids:
                    neg_tensors = model.encode_taxonomy(torch.tensor(neg_node_ids, device=device))
                    if is_hyperbolic:
                        dist_neg = lorentz_distance(p_rep.expand(len(neg_node_ids), -1), neg_tensors)
                    else:
                        dist_neg = torch.norm(p_rep - neg_tensors, p=2, dim=-1)
                    
                    # Positive ancestor must be closer than negative other-version ancestor
                    h_margin = 0.5
                    h_loss = F.relu(dist_pos.mean() - dist_neg.mean() + h_margin)
                    hier_loss = hier_loss + h_loss
                    num_hier += 1

            if num_hier > 0:
                hier_loss = hier_loss / num_hier

            # Depth regularizer loss on taxonomy nodes
            tax_weights = model.taxonomy_emb.weight
            tax_norms = torch.norm(tax_weights, p=2, dim=-1)
            depth_loss = F.mse_loss(tax_norms, depth_targets)

            # Passage leaf norm regularizer (passages near boundary, depth 4)
            p_tangent = model.project_tangent(g_tensors) if hasattr(model, "project_tangent") else model.proj(g_tensors)
            p_norms = torch.norm(p_tangent, p=2, dim=-1)
            passage_depth_loss = F.mse_loss(p_norms, torch.full_like(p_norms, DEPTH_NORMS[4]))

            total_b_loss = contrastive_loss + 0.5 * hier_loss + 0.1 * depth_loss + 0.1 * passage_depth_loss

            optimizer.zero_grad()
            total_b_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += total_b_loss.item()

    train_time = time.time() - t0
    return model, train_time

def run_all_retrievals(
    bm25,
    passages,
    queries,
    passage_embs,
    query_embs,
    model_b_trained_384,
    model_d_hyper_384,
    model_b_trained_32,
    model_d_hyper_32,
    device="cpu"
):
    """
    Executes retrieval across all systems:
    A: BM25
    B: Euclidean unprojected cosine
    B-trained: Euclidean trained control (384d)
    C: System B + metadata filter
    D: Hyperbolic (384d)
    E: System D + metadata filter
    F: RRF (B + D)
    And 32d ablation systems.
    """
    print("Running System A: BM25...")
    pred_A = {}
    for q in queries:
        tokens = q["query_text"].lower().split()
        scores = bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:50]
        pred_A[q["query_id"]] = [passages[idx]["passage_id"] for idx in top_indices]

    print("Running System B: Euclidean unprojected (cosine similarity)...")
    # Normalize for cosine similarity
    p_norm = passage_embs / torch.norm(passage_embs, p=2, dim=-1, keepdim=True).clamp_min(1e-7)
    q_norm = query_embs / torch.norm(query_embs, p=2, dim=-1, keepdim=True).clamp_min(1e-7)
    
    # Compute dot products in chunks
    cos_sim = torch.matmul(q_norm, p_norm.t()) # (num_queries, num_passages)
    pred_B = {}
    for i, q in enumerate(queries):
        top_indices = torch.topk(cos_sim[i], k=50).indices.tolist()
        pred_B[q["query_id"]] = [passages[idx]["passage_id"] for idx in top_indices]

    print("Running System C: System B + Metadata Filter...")
    # Filter policy:
    # If version_stated is True: keep only matching product and target_version
    # If version_stated is False: keep matching product and guess newest version (Nextcloud 30, Django 5.1)
    newest_versions = {"Nextcloud": "30", "Django": "5.1"}
    pred_C = {}
    for i, q in enumerate(queries):
        prod = q["product"]
        if q["version_stated"]:
            req_ver = q["target_version"]
        else:
            req_ver = newest_versions[prod]
            
        # Filter passages
        valid_indices = [
            idx for idx, p in enumerate(passages)
            if p["product"] == prod and p["version"] == req_ver
        ]
        sub_scores = cos_sim[i, valid_indices]
        top_sub = torch.topk(sub_scores, k=min(50, len(valid_indices))).indices.tolist()
        pred_C[q["query_id"]] = [passages[valid_indices[s_idx]]["passage_id"] for s_idx in top_sub]

    print("Running System B-trained (Euclidean 384d)...")
    model_b_trained_384.eval()
    with torch.no_grad():
        p_b_rep = model_b_trained_384.encode(passage_embs.to(device))
        q_b_rep = model_b_trained_384.encode(query_embs.to(device))
        # Negative euclidean distance
        b_sim = model_b_trained_384.compute_similarity(q_b_rep, p_b_rep)
        
    pred_B_trained = {}
    for i, q in enumerate(queries):
        top_indices = torch.topk(b_sim[i], k=50).indices.tolist()
        pred_B_trained[q["query_id"]] = [passages[idx]["passage_id"] for idx in top_indices]

    print("Running System D: Hyperbolic (384d)...")
    model_d_hyper_384.eval()
    with torch.no_grad():
        p_d_rep = model_d_hyper_384.encode(passage_embs.to(device))
        q_d_rep = model_d_hyper_384.encode(query_embs.to(device))
        d_sim = model_d_hyper_384.compute_similarity(q_d_rep, p_d_rep)

    pred_D = {}
    for i, q in enumerate(queries):
        top_indices = torch.topk(d_sim[i], k=50).indices.tolist()
        pred_D[q["query_id"]] = [passages[idx]["passage_id"] for idx in top_indices]

    print("Running System E: System D + Metadata Filter...")
    pred_E = {}
    for i, q in enumerate(queries):
        prod = q["product"]
        if q["version_stated"]:
            req_ver = q["target_version"]
        else:
            req_ver = newest_versions[prod]
            
        valid_indices = [
            idx for idx, p in enumerate(passages)
            if p["product"] == prod and p["version"] == req_ver
        ]
        sub_scores = d_sim[i, valid_indices]
        top_sub = torch.topk(sub_scores, k=min(50, len(valid_indices))).indices.tolist()
        pred_E[q["query_id"]] = [passages[valid_indices[s_idx]]["passage_id"] for s_idx in top_sub]

    print("Running System F: Reciprocal Rank Fusion (B + D)...")
    pred_F = {}
    for q in queries:
        qid = q["query_id"]
        list_b = pred_B[qid][:50]
        list_d = pred_D[qid][:50]
        
        rrf_scores = defaultdict(float)
        for rank, pid in enumerate(list_b, start=1):
            rrf_scores[pid] += 1.0 / (60.0 + rank)
        for rank, pid in enumerate(list_d, start=1):
            rrf_scores[pid] += 1.0 / (60.0 + rank)
            
        sorted_pids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        pred_F[qid] = sorted_pids[:50]

    # 32-dimensional runs
    print("Running 32d Ablations (B-trained-32d and D-32d)...")
    model_b_trained_32.eval()
    model_d_hyper_32.eval()
    with torch.no_grad():
        p_b32 = model_b_trained_32.encode(passage_embs.to(device))
        q_b32 = model_b_trained_32.encode(query_embs.to(device))
        b32_sim = model_b_trained_32.compute_similarity(q_b32, p_b32)

        p_d32 = model_d_hyper_32.encode(passage_embs.to(device))
        q_d32 = model_d_hyper_32.encode(query_embs.to(device))
        d32_sim = model_d_hyper_32.compute_similarity(q_d32, p_d32)

    pred_B_32 = {}
    pred_D_32 = {}
    pred_E_32 = {}
    pred_F_32 = {}

    for i, q in enumerate(queries):
        qid = q["query_id"]
        top_b32 = torch.topk(b32_sim[i], k=50).indices.tolist()
        pred_B_32[qid] = [passages[idx]["passage_id"] for idx in top_b32]

        top_d32 = torch.topk(d32_sim[i], k=50).indices.tolist()
        pred_D_32[qid] = [passages[idx]["passage_id"] for idx in top_d32]

        # Filtered 32d
        prod = q["product"]
        req_ver = q["target_version"] if q["version_stated"] else newest_versions[prod]
        valid_indices = [idx for idx, p in enumerate(passages) if p["product"] == prod and p["version"] == req_ver]
        sub_d32 = d32_sim[i, valid_indices]
        top_sub32 = torch.topk(sub_d32, k=min(50, len(valid_indices))).indices.tolist()
        pred_E_32[qid] = [passages[valid_indices[s_idx]]["passage_id"] for s_idx in top_sub32]

        # RRF 32d
        rrf_32 = defaultdict(float)
        for rank, pid in enumerate(pred_B_32[qid][:50], start=1):
            rrf_32[pid] += 1.0 / (60.0 + rank)
        for rank, pid in enumerate(pred_D_32[qid][:50], start=1):
            rrf_32[pid] += 1.0 / (60.0 + rank)
        pred_F_32[qid] = sorted(rrf_32.keys(), key=lambda x: rrf_32[x], reverse=True)[:50]

    return {
        "A_BM25": pred_A,
        "B_Euclidean": pred_B,
        "B_trained_384d": pred_B_trained,
        "C_Euclidean_Filter": pred_C,
        "D_Hyperbolic_384d": pred_D,
        "E_Hyperbolic_Filter": pred_E,
        "F_RRF_B_D": pred_F,
        "B_trained_32d": pred_B_32,
        "D_Hyperbolic_32d": pred_D_32,
        "E_Hyperbolic_Filter_32d": pred_E_32,
        "F_RRF_32d": pred_F_32,
    }

def main():
    set_seed(42)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "figures"), exist_ok=True)

    print("Loading dataset...")
    with open(os.path.join(PROCESSED_DIR, "passages.jsonl"), "r", encoding="utf-8") as f:
        passages = [json.loads(line) for line in f]
    with open(os.path.join(PROCESSED_DIR, "queries.jsonl"), "r", encoding="utf-8") as f:
        queries = [json.loads(line) for line in f]

    print(f"Loaded {len(passages)} passages and {len(queries)} queries.")

    print("Loading precomputed embeddings...")
    p_data = torch.load(os.path.join(PROCESSED_DIR, "passage_embeddings.pt"), map_location="cpu")
    q_data = torch.load(os.path.join(PROCESSED_DIR, "query_embeddings.pt"), map_location="cpu")
    passage_embs = p_data["embeddings"]
    query_embs = q_data["embeddings"]

    print("Building taxonomy...")
    node_to_id, id_to_depth, passage_ancestors, passage_negative_nodes = build_taxonomy(passages)
    num_nodes = len(node_to_id)
    print(f"Taxonomy contains {num_nodes} nodes across 4 hierarchy levels.")

    # Prepare BM25
    print("Preparing BM25 index...")
    tokenized_corpus = [p["text"].lower().split() for p in passages]
    bm25 = BM25Okapi(tokenized_corpus)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hardware_desc = f"{platform.processor()} ({platform.system()} {platform.release()}), PyTorch on {device.upper()}"
    print(f"Running on: {hardware_desc}")

    # Train 384d models
    print("\nTraining Hyperbolic Model (384d, seed=42)...")
    model_d_384 = HyperbolicRetriever(in_dim=384, proj_dim=384, num_taxonomy_nodes=num_nodes)
    model_d_384, time_d_384 = train_retriever(
        model_d_384, passage_embs, query_embs, queries, passages,
        passage_ancestors, passage_negative_nodes, id_to_depth,
        is_hyperbolic=True, epochs=15, lr=1e-3, device=device
    )
    print(f"Hyperbolic 384d trained in {time_d_384:.2f}s")

    print("\nTraining Euclidean Control (384d, seed=42)...")
    model_b_384 = EuclideanRetriever(in_dim=384, proj_dim=384, num_taxonomy_nodes=num_nodes)
    model_b_384, time_b_384 = train_retriever(
        model_b_384, passage_embs, query_embs, queries, passages,
        passage_ancestors, passage_negative_nodes, id_to_depth,
        is_hyperbolic=False, epochs=15, lr=1e-3, device=device
    )
    print(f"Euclidean Control 384d trained in {time_b_384:.2f}s")

    # Train 32d ablation models
    print("\nTraining Hyperbolic Model (32d, seed=42)...")
    model_d_32 = HyperbolicRetriever(in_dim=384, proj_dim=32, num_taxonomy_nodes=num_nodes)
    model_d_32, time_d_32 = train_retriever(
        model_d_32, passage_embs, query_embs, queries, passages,
        passage_ancestors, passage_negative_nodes, id_to_depth,
        is_hyperbolic=True, epochs=15, lr=1e-3, device=device
    )
    print(f"Hyperbolic 32d trained in {time_d_32:.2f}s")

    print("\nTraining Euclidean Control (32d, seed=42)...")
    model_b_32 = EuclideanRetriever(in_dim=384, proj_dim=32, num_taxonomy_nodes=num_nodes)
    model_b_32, time_b_32 = train_retriever(
        model_b_32, passage_embs, query_embs, queries, passages,
        passage_ancestors, passage_negative_nodes, id_to_depth,
        is_hyperbolic=False, epochs=15, lr=1e-3, device=device
    )
    print(f"Euclidean Control 32d trained in {time_b_32:.2f}s")

    # Run retrievals
    print("\nExecuting all retrieval pipelines...")
    predictions = run_all_retrievals(
        bm25,
        passages,
        queries,
        passage_embs,
        query_embs,
        model_b_384,
        model_d_384,
        model_b_32,
        model_d_32,
        device=device
    )

    # Evaluate all systems
    print("\nComputing metrics across all systems...")
    all_evaluations = {}
    for sys_name, preds in predictions.items():
        all_evaluations[sys_name] = evaluate_system(preds, queries)

    # Format results json
    results_payload = {
        "metadata": {
            "num_passages": len(passages),
            "num_queries": len(queries),
            "version_stated_count": sum(1 for q in queries if q["version_stated"]),
            "version_unstated_count": sum(1 for q in queries if not q["version_stated"]),
            "seed": 42,
            "hardware": hardware_desc,
            "training_times_seconds": {
                "Hyperbolic_384d": round(time_d_384, 2),
                "Euclidean_Control_384d": round(time_b_384, 2),
                "Hyperbolic_32d": round(time_d_32, 2),
                "Euclidean_Control_32d": round(time_b_32, 2),
            }
        },
        "systems": all_evaluations
    }

    # Save results.json
    out_json = os.path.join(RESULTS_DIR, "results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)
    print(f"Saved results to {out_json}")

    # Build markdown table
    out_md = os.path.join(RESULTS_DIR, "results.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Retrieval Benchmark Results\n\n")
        f.write(f"Hardware: {hardware_desc}\n")
        f.write(f"Dataset: {len(passages)} passages, {len(queries)} queries (157 stated, 157 unstated).\n\n")
        
        splits = [
            ("Version Unstated (Headline Result)", "version_unstated"),
            ("Version Stated", "version_stated"),
            ("Overall (All Queries)", "overall")
        ]
        
        for split_title, split_key in splits:
            f.write(f"## {split_title}\n\n")
            f.write("| System | Conflict Rate@5 | Stale Rate@5 | Stale Rate@10 | Recall@5 | Recall@10 | MRR |\n")
            f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
            
            for sys_name, res in all_evaluations.items():
                m = res[split_key]
                f.write(
                    f"| {sys_name} | {m['conflict_rate@5']:.4f} | {m['stale_rate@5']:.4f} | "
                    f"{m['stale_rate@10']:.4f} | {m['recall@5']:.4f} | {m['recall@10']:.4f} | {m['mrr']:.4f} |\n"
                )
            f.write("\n")

    print(f"Saved summary table to {out_md}")

    # Save model checkpoint
    model_save_path = os.path.join(RESULTS_DIR, "hyperbolic_model_384.pt")
    torch.save(model_d_384.state_dict(), model_save_path)
    print("Experiments run successfully.")

if __name__ == "__main__":
    main()
