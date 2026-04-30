"""Tests for retrieval evaluation metrics."""

from unittest.mock import MagicMock

from ravenrag.eval import EvalResult, _ndcg, _recall_at_k, _reciprocal_rank, evaluate
from ravenrag.index import QueryResult


class TestReciprocalRank:
    def test_first_result_relevant(self):
        assert _reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0

    def test_second_result_relevant(self):
        assert _reciprocal_rank(["a", "b", "c"], {"b"}) == 0.5

    def test_third_result_relevant(self):
        assert abs(_reciprocal_rank(["a", "b", "c"], {"c"}) - 1.0 / 3) < 1e-9

    def test_no_relevant(self):
        assert _reciprocal_rank(["a", "b", "c"], {"x"}) == 0.0

    def test_empty_retrieved(self):
        assert _reciprocal_rank([], {"a"}) == 0.0


class TestNDCG:
    def test_perfect_ranking(self):
        assert _ndcg(["a", "b"], {"a", "b"}) == 1.0

    def test_no_relevant(self):
        assert _ndcg(["a", "b"], {"x"}) == 0.0

    def test_partial_relevant(self):
        val = _ndcg(["a", "b", "c"], {"b"})
        assert 0.0 < val < 1.0


class TestRecallAtK:
    def test_full_recall(self):
        assert _recall_at_k(["a", "b"], {"a", "b"}) == 1.0

    def test_partial_recall(self):
        assert _recall_at_k(["a", "c"], {"a", "b"}) == 0.5

    def test_no_recall(self):
        assert _recall_at_k(["x", "y"], {"a", "b"}) == 0.0

    def test_empty_relevant(self):
        assert _recall_at_k(["a", "b"], set()) == 0.0


class TestEvaluate:
    def test_perfect_retrieval(self):
        index = MagicMock()
        index.query.return_value = [
            QueryResult(id="d1", text="text", metadata={}, distance=0.1),
        ]

        result = evaluate(
            index,
            queries=["q1"],
            expected_ids=[["d1"]],
            top_k=1,
        )
        assert isinstance(result, EvalResult)
        assert result.mrr == 1.0
        assert result.recall == 1.0
        assert result.ndcg == 1.0
        assert len(result.per_query) == 1

    def test_no_match(self):
        index = MagicMock()
        index.query.return_value = [
            QueryResult(id="d2", text="text", metadata={}, distance=0.5),
        ]

        result = evaluate(
            index,
            queries=["q1"],
            expected_ids=[["d1"]],
            top_k=1,
        )
        assert result.mrr == 0.0
        assert result.recall == 0.0

    def test_multiple_queries(self):
        index = MagicMock()
        index.query.side_effect = [
            [QueryResult(id="d1", text="t", metadata={}, distance=0.1)],
            [QueryResult(id="d3", text="t", metadata={}, distance=0.2)],
        ]

        result = evaluate(
            index,
            queries=["q1", "q2"],
            expected_ids=[["d1"], ["d3"]],
            top_k=1,
        )
        assert result.mrr == 1.0
        assert len(result.per_query) == 2

    def test_mismatched_lengths_raises(self):
        index = MagicMock()
        import pytest

        with pytest.raises(ValueError, match="same length"):
            evaluate(index, queries=["q1"], expected_ids=[["d1"], ["d2"]])
