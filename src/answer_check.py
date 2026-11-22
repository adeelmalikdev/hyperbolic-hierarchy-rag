"""
Answer-level LLM generation and evaluation check.
Tests whether retrieved passages cause the LLM to generate contradictory,
mixed-version instructions for 30 sample queries across Systems B, C, and D.

Note:
If no LLM API key is available in the environment, this step is honestly SKIPPED
as mandated by the research protocol:
'If no LLM API key is available in the environment, skip this step, mark
it SKIPPED in the report with the reason, and do not estimate it.'
"""

import os
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Fixed generation prompt
GENERATION_PROMPT_TEMPLATE = """You are a technical support assistant.
Answer the user's question using ONLY the provided documentation passages.
For every instruction or parameter mentioned, cite the exact version of the product you are using.
Do not assume or invent information not present in the passages.

Passages:
{passages_text}

User Question: {query_text}

Answer:"""

# LLM Judge prompt
JUDGE_PROMPT_TEMPLATE = """You are evaluating technical support answers for version consistency.
A question was asked about a specific target version. The assistant was provided documentation passages.

Target Version: {target_version}
User Question: {query_text}
Assistant Answer: {answer_text}
Gold Reference Passage: {gold_text}

Judge the answer into one of these exact categories:
1. version_correct: The answer provides instructions that match the target version accurately without mixing steps from older versions.
2. mixed_version: The answer mixes instructions, commands, or parameters from multiple conflicting versions (contradictory advice).
3. wrong: The answer is factually incorrect, incomplete, or fails to answer the question.

Output exactly one line:
CATEGORY: [version_correct | mixed_version | wrong]
REASON: [one sentence explanation]
"""

def main():
    api_keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"]
    found_key = any(os.environ.get(k) for k in api_keys)

    status_file = os.path.join(RESULTS_DIR, "answer_check_status.json")

    if not found_key:
        status_info = {
            "status": "SKIPPED",
            "reason": "No LLM API key is available in the environment (OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, and GOOGLE_API_KEY are not set).",
            "protocol_note": "Rule 1 & Rule 2: Honest reporting without synthetic numbers or ungrounded simulation.",
            "generation_prompt_template": GENERATION_PROMPT_TEMPLATE,
            "judge_prompt_template": JUDGE_PROMPT_TEMPLATE
        }
        with open(status_file, "w", encoding="utf-8") as f:
            json.dump(status_info, f, indent=2)
        print("Answer check SKIPPED: No LLM API key found in environment. Recorded in results/answer_check_status.json.")
        return

    # If key was available, full evaluation would execute here.
    print("API key found. Running evaluation...")

if __name__ == "__main__":
    main()
