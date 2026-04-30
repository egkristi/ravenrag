"""Tests for the timing/observability module."""

from ravenrag.timing import get_timings, reset_timings, timed


class TestTiming:
    def setup_method(self):
        reset_timings()

    def test_timed_decorator(self):
        @timed("test_op")
        def dummy():
            return 42

        result = dummy()
        assert result == 42
        timings = get_timings()
        assert "test_op" in timings
        assert timings["test_op"]["calls"] == 1
        assert timings["test_op"]["total_seconds"] >= 0

    def test_timed_accumulates(self):
        @timed("accum")
        def dummy():
            pass

        dummy()
        dummy()
        dummy()
        assert get_timings()["accum"]["calls"] == 3

    def test_reset_timings(self):
        @timed("reset_test")
        def dummy():
            pass

        dummy()
        assert "reset_test" in get_timings()
        reset_timings()
        assert get_timings() == {}

    def test_timed_default_name(self):
        @timed()
        def my_function():
            pass

        my_function()
        timings = get_timings()
        assert any("my_function" in k for k in timings)
