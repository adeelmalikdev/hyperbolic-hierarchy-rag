"""
Metrics computation module for version-conflicted retrieval evaluation.
Primary metrics:
- stale_passage_rate@K: share of retrieved passages in top-K that are stale passages for the query issue.
- conflict_rate@5: share of queries whose top 5 contains BOTH the gold passage and at least one stale passage.
Secondary metrics:
- Recall@K: whether the gold passage appears in top-K.
- MRR: reciprocal rank of the gold passage.
"""

from typing import List, Dict, Any

def compute_query_metrics(retrieved_pids: List[str], gold_pid: str, stale_pids: List[str]) -> Dict[str, float]:
    """
    Computes all evaluation metrics for a single query.
    """
    stale_set = set(stale_pids)
    
    top5 = retrieved_pids[:5]
    top10 = retrieved_pids[:10]
    
    # stale_passage_rate@K
    stale_count_5 = sum(1 for pid in top5 if pid in stale_set)
    stale_rate_5 = stale_count_5 / 5.0 if top5 else 0.0
    
    stale_count_10 = sum(1 for pid in top10 if pid in stale_set)
    stale_rate_10 = stale_count_10 / float(len(top10)) if top10 else 0.0
    
    # conflict_rate@5: contains BOTH gold and at least one stale passage in top 5
    has_gold_in_top5 = gold_pid in top5
    has_stale_in_top5 = any(pid in stale_set for pid in top5)
    conflict_5 = 1.0 if (has_gold_in_top5 and has_stale_in_top5) else 0.0
    
    # Recall@K
    recall_5 = 1.0 if has_gold_in_top5 else 0.0
    recall_10 = 1.0 if (gold_pid in top10) else 0.0
    
    # MRR (up to retrieved length)
    mrr = 0.0
    for rank, pid in enumerate(retrieved_pids, start=1):
        if pid == gold_pid:
            mrr = 1.0 / rank
            break
            
    return {
        "stale_rate@5": stale_rate_5,
        "stale_rate@10": stale_rate_10,
        "conflict_rate@5": conflict_5,
        "recall@5": recall_5,
        "recall@10": recall_10,
        "mrr": mrr,
    }

def aggregate_metrics(query_results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Aggregates list of per-query metric dicts into average metrics.
    """
    if not query_results:
        return {
            "stale_rate@5": 0.0,
            "stale_rate@10": 0.0,
            "conflict_rate@5": 0.0,
            "recall@5": 0.0,
            "recall@10": 0.0,
            "mrr": 0.0,
            "count": 0,
        }
        
    n = len(query_results)
    keys = ["stale_rate@5", "stale_rate@10", "conflict_rate@5", "recall@5", "recall@10", "mrr"]
    agg = {k: sum(q[k] for q in query_results) / float(n) for k in keys}
    agg["count"] = n
    return agg

def evaluate_system(predictions: Dict[str, List[str]], queries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluates predictions for a set of queries and splits by version_stated.
    Returns:
      overall, stated, unstated
    """
    all_metrics = []
    stated_metrics = []
    unstated_metrics = []
    
    for q in queries:
        qid = q["query_id"]
        retrieved = predictions.get(qid, [])
        m = compute_query_metrics(retrieved, q["gold_passage_id"], q.get("stale_passage_ids", []))
        all_metrics.append(m)
        if q.get("version_stated", False):
            stated_metrics.append(m)
        else:
            unstated_metrics.append(m)
            
    return {
        "overall": aggregate_metrics(all_metrics),
        "version_stated": aggregate_metrics(stated_metrics),
        "version_unstated": aggregate_metrics(unstated_metrics),
    }
