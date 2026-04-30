"""
Eval: Retrieval quality metrics for RavenRAG.

Compute MRR, NDCG, and Recall@k to quantify how well your
index retrieves the right documents.

Example::

    from ravenrag.eval import evaluate

    results = evaluate(
        index,
        queries=["What is RAG?", "How to chunk?"],
        expected_ids=[["doc1", "doc3"], ["doc7"]],
    )
    print(results.mrr, results.ndcg, results.recall)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from .index import DocumentIndex


@dataclass
class EvalResult:
    """Aggregated retrieval quality metrics."""

    mrr: float
    """Mean Reciprocal Rank — how early the first relevant result appears."""

    ndcg: float
    """Normalized Discounted Cumulative Gain — ranking quality."""

    recall: float
    """Recall@k — fraction of relevant documents retrieved."""

    per_query: List[Dict[str, float]]
    """Per-query breakdown."""


def _reciprocal_rank(retrieved_ids: List[str], relevant_ids: set) -> float:
    """RR for a single query."""
    for i, rid in enumerate(retrieved_ids, 1):
        if rid in relevant_ids:
            return 1.0 / i
    return 0.0


def _dcg(relevances: List[float]) -> float:
    """Discounted Cumulative Gain."""
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))


def _ndcg(retrieved_ids: List[str], relevant_ids: set) -> float:
    """NDCG for a single query."""
    relevances = [1.0 if rid in relevant_ids else 0.0 for rid in retrieved_ids]
    actual_dcg = _dcg(relevances)
    # Ideal: all relevant docs at top
    ideal = sorted(relevances, reverse=True)
    ideal_dcg = _dcg(ideal)
    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


def _recall_at_k(retrieved_ids: List[str], relevant_ids: set) -> float:
    """Recall@k for a single query."""
    if not relevant_ids:
        return 0.0
    found = sum(1 for rid in retrieved_ids if rid in relevant_ids)
    return found / len(relevant_ids)


def evaluate(
    index: DocumentIndex,
    queries: List[str],
    expected_ids: List[List[str]],
    top_k: int = 5,
    where: Optional[Dict] = None,
) -> EvalResult:
    """Evaluate retrieval quality against ground truth.

    Args:
        index: A DocumentIndex to query.
        queries: List of test queries.
        expected_ids: For each query, a list of document IDs that are relevant.
        top_k: Number of results to retrieve per query.
        where: Optional metadata filter.

    Returns:
        EvalResult with aggregated and per-query metrics.
    """
    if len(queries) != len(expected_ids):
        raise ValueError("queries and expected_ids must have the same length")

    per_query: List[Dict[str, float]] = []
    total_mrr = 0.0
    total_ndcg = 0.0
    total_recall = 0.0

    for query, expected in zip(queries, expected_ids):
        results = index.query(query, top_k=top_k, where=where)
        retrieved = [r.id for r in results]
        relevant = set(expected)

        rr = _reciprocal_rank(retrieved, relevant)
        ndcg_val = _ndcg(retrieved, relevant)
        recall_val = _recall_at_k(retrieved, relevant)

        per_query.append({"query": query, "mrr": rr, "ndcg": ndcg_val, "recall": recall_val})
        total_mrr += rr
        total_ndcg += ndcg_val
        total_recall += recall_val

    n = len(queries) or 1
    return EvalResult(
        mrr=total_mrr / n,
        ndcg=total_ndcg / n,
        recall=total_recall / n,
        per_query=per_query,
    )
