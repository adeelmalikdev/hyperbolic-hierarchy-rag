"""
Precomputes and caches SentenceTransformer all-MiniLM-L6-v2 embeddings
for all passages and queries.
"""

import os
import time
import json
import torch
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SentenceTransformer all-MiniLM-L6-v2 on {device}...")
    t0 = time.time()
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    print(f"Model loaded in {time.time() - t0:.2f}s")

    # Load passages
    passages_file = os.path.join(PROCESSED_DIR, "passages.jsonl")
    queries_file = os.path.join(PROCESSED_DIR, "queries.jsonl")

    print("Reading passages...")
    passage_ids = []
    passage_texts = []
    with open(passages_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            passage_ids.append(item["passage_id"])
            passage_texts.append(item["text"])

    print(f"Total passages to encode: {len(passage_texts)}")
    t1 = time.time()
    passage_embs = model.encode(
        passage_texts,
        batch_size=128,
        show_progress_bar=True,
        convert_to_tensor=True,
        device=device
    )
    t2 = time.time()
    print(f"Encoded {len(passage_texts)} passages in {t2 - t1:.2f}s ({len(passage_texts)/(t2 - t1):.1f} passages/sec)")

    print("Reading queries...")
    query_ids = []
    query_texts = []
    with open(queries_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            query_ids.append(item["query_id"])
            query_texts.append(item["query_text"])

    print(f"Total queries to encode: {len(query_texts)}")
    query_embs = model.encode(
        query_texts,
        batch_size=64,
        show_progress_bar=False,
        convert_to_tensor=True,
        device=device
    )

    # Save tensors and id maps
    out_passages = os.path.join(PROCESSED_DIR, "passage_embeddings.pt")
    out_queries = os.path.join(PROCESSED_DIR, "query_embeddings.pt")

    torch.save({"ids": passage_ids, "embeddings": passage_embs.cpu()}, out_passages)
    torch.save({"ids": query_ids, "embeddings": query_embs.cpu()}, out_queries)

    print(f"Saved passage embeddings to {out_passages}")
    print(f"Saved query embeddings to {out_queries}")

if __name__ == "__main__":
    main()
