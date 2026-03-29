"""Tests for probe — covers scalars, collections, tensors, config, and output."""

import io
import re
import sys
import pytest
import numpy as np

from probe_inspect import probe, pb, probe_config, ProbeConfig


@pytest.fixture(autouse=True)
def _capture_to_buffer(monkeypatch):
    """Redirect probe output to a buffer for assertions."""
    buf = io.StringIO()
    monkeypatch.setattr(probe_config, "output", buf)
    monkeypatch.setattr(probe_config, "color", False)
    yield buf


def last_line(buf: io.StringIO) -> str:
    return buf.getvalue().strip().split("\n")[-1]


# ─── Basic Values ────────────────────────────────────────────────────────────

class TestScalars:
    def test_int(self, _capture_to_buffer):
        probe(42)
        assert "42" in last_line(_capture_to_buffer)
        assert "(int)" in last_line(_capture_to_buffer)

    def test_float(self, _capture_to_buffer):
        probe(3.14)
        assert "3.14" in last_line(_capture_to_buffer)
        assert "(float)" in last_line(_capture_to_buffer)

    def test_string(self, _capture_to_buffer):
        probe("hello world")
        assert "'hello world'" in last_line(_capture_to_buffer)
        assert "(str)" in last_line(_capture_to_buffer)

    def test_bool(self, _capture_to_buffer):
        probe(True)
        assert "True" in last_line(_capture_to_buffer)
        assert "(bool)" in last_line(_capture_to_buffer)

    def test_none(self, _capture_to_buffer):
        probe(None)
        assert "None" in last_line(_capture_to_buffer)
        assert "(NoneType)" in last_line(_capture_to_buffer)

    def test_complex(self, _capture_to_buffer):
        probe(1 + 2j)
        assert "(1+2j)" in last_line(_capture_to_buffer)


# ─── Collections ─────────────────────────────────────────────────────────────

class TestCollections:
    def test_list(self, _capture_to_buffer):
        probe([1, 2, 3])
        out = _capture_to_buffer.getvalue()
        assert "(list)" in out
        assert "1" in out
        assert "3" in out

    def test_dict(self, _capture_to_buffer):
        probe({"a": 1, "b": 2})
        out = _capture_to_buffer.getvalue()
        assert "(dict)" in out
        assert "'a'" in out

    def test_tuple(self, _capture_to_buffer):
        probe((10, 20))
        out = _capture_to_buffer.getvalue()
        assert "(tuple)" in out
        assert "10" in out

    def test_empty_list(self, _capture_to_buffer):
        probe([])
        assert "[]" in last_line(_capture_to_buffer)

    def test_nested(self, _capture_to_buffer):
        probe({"layers": [512, 1024, 512]})
        out = _capture_to_buffer.getvalue()
        assert "512" in out
        assert "1024" in out

    def test_compact(self, _capture_to_buffer):
        probe({"a": 1, "b": 2}, compact=True)
        # Compact should be single line
        lines = _capture_to_buffer.getvalue().strip().split("\n")
        assert len(lines) == 1


# ─── Numpy Tensors ───────────────────────────────────────────────────────────

class TestNumpy:
    def test_shape_and_dtype(self, _capture_to_buffer):
        x = np.zeros((8, 64, 512), dtype=np.float32)
        probe(x)
        out = last_line(_capture_to_buffer)
        assert "float32" in out
        assert "(8, 64, 512)" in out

    def test_stats(self, _capture_to_buffer):
        x = np.random.randn(100).astype(np.float32)
        probe(x)
        out = last_line(_capture_to_buffer)
        assert "μ=" in out
        assert "σ=" in out
        assert "∈[" in out

    def test_nan_warning(self, _capture_to_buffer):
        x = np.array([1.0, float("nan"), 3.0])
        probe(x)
        out = last_line(_capture_to_buffer)
        assert "1 NaN" in out

    def test_inf_warning(self, _capture_to_buffer):
        x = np.array([1.0, float("inf"), -float("inf")])
        probe(x)
        out = last_line(_capture_to_buffer)
        assert "2 Inf" in out

    def test_param_count(self, _capture_to_buffer):
        x = np.zeros((1000, 1000), dtype=np.float32)
        probe(x)
        assert "1.00M params" in last_line(_capture_to_buffer)

    def test_param_count_K(self, _capture_to_buffer):
        x = np.zeros((100, 100), dtype=np.float32)
        probe(x)
        assert "10.0K params" in last_line(_capture_to_buffer)

    def test_scalar_tensor(self, _capture_to_buffer):
        x = np.float32(42.0)
        probe(x)
        out = last_line(_capture_to_buffer)
        assert "float32" in out


# ─── Multiple Values ─────────────────────────────────────────────────────────

class TestMultipleValues:
    def test_two_values(self, _capture_to_buffer):
        probe(42, "hello")
        out = _capture_to_buffer.getvalue()
        assert "42" in out
        assert "'hello'" in out

    def test_mixed_tensor_scalar(self, _capture_to_buffer):
        x = np.zeros((4, 8))
        probe(x, 42)
        out = _capture_to_buffer.getvalue()
        assert "(4, 8)" in out
        assert "42" in out


# ─── Pass-through Return ─────────────────────────────────────────────────────

class TestReturn:
    def test_single_returns_value(self, _capture_to_buffer):
        result = probe(42)
        assert result == 42

    def test_single_returns_tensor(self, _capture_to_buffer):
        x = np.zeros((2, 3))
        result = probe(x)
        assert result is x

    def test_multiple_returns_tuple(self, _capture_to_buffer):
        result = probe(1, 2, 3)
        assert result == (1, 2, 3)


# ─── Tags ────────────────────────────────────────────────────────────────────

class TestTags:
    def test_tag_in_output(self, _capture_to_buffer):
        probe(42, tag="ATTN")
        assert "ATTN" in last_line(_capture_to_buffer)

    def test_global_tag(self, _capture_to_buffer, monkeypatch):
        monkeypatch.setattr(probe_config, "tag", "GLOBAL")
        probe(42)
        assert "GLOBAL" in last_line(_capture_to_buffer)

    def test_per_call_overrides_global(self, _capture_to_buffer, monkeypatch):
        monkeypatch.setattr(probe_config, "tag", "GLOBAL")
        probe(42, tag="LOCAL")
        out = last_line(_capture_to_buffer)
        assert "LOCAL" in out


# ─── Config ──────────────────────────────────────────────────────────────────

class TestConfig:
    def test_disabled(self, _capture_to_buffer, monkeypatch):
        monkeypatch.setattr(probe_config, "enabled", False)
        result = probe(42)
        assert result == 42
        assert _capture_to_buffer.getvalue() == ""

    def test_no_type(self, _capture_to_buffer):
        probe(42, show_type=False)
        assert "(int)" not in last_line(_capture_to_buffer)

    def test_no_location(self, _capture_to_buffer):
        probe(42, show_loc=False)
        # Should not contain the [file:line] bracket
        out = last_line(_capture_to_buffer)
        assert "[" not in out or "test_" not in out

    def test_no_stats(self, _capture_to_buffer):
        x = np.random.randn(100).astype(np.float32)
        probe(x, show_stats=False)
        assert "μ=" not in last_line(_capture_to_buffer)

    def test_timestamp(self, _capture_to_buffer):
        probe(42, show_time=True)
        out = last_line(_capture_to_buffer)
        assert ":" in out.split("[")[0]  # time before location

    def test_stat_fmt(self, _capture_to_buffer):
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        probe(x, stat_fmt=".2f")
        out = last_line(_capture_to_buffer)
        # With .2f, mean of [1,2,3] = 2.00
        assert "2.00" in out


# ─── Location ────────────────────────────────────────────────────────────────

class TestLocation:
    def test_shows_filename(self, _capture_to_buffer):
        probe(42)
        assert "test_probe.py" in last_line(_capture_to_buffer)

    def test_shows_function_name(self, _capture_to_buffer):
        def my_custom_function():
            probe(42)
        my_custom_function()
        assert "my_custom_function()" in last_line(_capture_to_buffer)

    def test_shows_line_number(self, _capture_to_buffer):
        probe(42)
        out = last_line(_capture_to_buffer)
        # Should have file:NUMBER pattern
        assert re.search(r":\d+", out)


# ─── Edge Cases ──────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_long_string(self, _capture_to_buffer):
        probe("a" * 500)
        out = _capture_to_buffer.getvalue()
        assert "..." in out

    def test_deeply_nested(self, _capture_to_buffer):
        val = {"a": {"b": {"c": {"d": {"e": "deep"}}}}}
        probe(val)
        out = _capture_to_buffer.getvalue()
        assert "..." in out  # Should hit depth limit

    def test_empty_dict(self, _capture_to_buffer):
        probe({})
        assert "{}" in last_line(_capture_to_buffer)

    def test_bytes(self, _capture_to_buffer):
        probe(b"hello")
        assert "b'hello'" in last_line(_capture_to_buffer)

    def test_custom_object(self, _capture_to_buffer):
        class Foo:
            def __repr__(self):
                return "Foo(bar=42)"
        probe(Foo())
        assert "Foo(bar=42)" in last_line(_capture_to_buffer)

    def test_all_nan_tensor(self, _capture_to_buffer):
        x = np.array([float("nan"), float("nan")])
        probe(x)
        assert "all NaN/Inf!" in last_line(_capture_to_buffer)


# ─── pb alias ────────────────────────────────────────────────────────────────

class TestPbAlias:
    def test_pb_is_probe(self):
        assert pb is probe

    def test_pb_works(self, _capture_to_buffer):
        pb(42)
        assert "42" in last_line(_capture_to_buffer)
        assert "(int)" in last_line(_capture_to_buffer)

    def test_pb_tensor(self, _capture_to_buffer):
        x = np.ones((4, 8), dtype=np.float32)
        pb(x)
        out = last_line(_capture_to_buffer)
        assert "float32" in out
        assert "(4, 8)" in out

    def test_pb_returns_value(self, _capture_to_buffer):
        result = pb(42)
        assert result == 42

    def test_pb_watch(self, _capture_to_buffer):
        @pb.watch
        def add(a, b):
            return a + b

        result = add(3, 4)
        assert result == 7
        out = _capture_to_buffer.getvalue()
        assert "add" in out
        assert "called" in out

    def test_pb_diff(self, _capture_to_buffer):
        a = np.ones((4, 4), dtype=np.float32)
        pb.diff(a, a)
        out = _capture_to_buffer.getvalue()
        assert "identical" in out


# ─── @probe.watch Decorator ─────────────────────────────────────────────────

class TestWatch:
    def test_watch_basic(self, _capture_to_buffer):
        @probe.watch
        def add(a, b):
            return a + b

        result = add(3, 4)
        assert result == 7
        out = _capture_to_buffer.getvalue()
        assert "add" in out
        assert "called" in out
        assert "done" in out
        assert "7" in out

    def test_watch_with_tag(self, _capture_to_buffer):
        @probe.watch(tag="MATH")
        def multiply(a, b):
            return a * b

        multiply(3, 4)
        out = _capture_to_buffer.getvalue()
        assert "MATH" in out

    def test_watch_tensor_args(self, _capture_to_buffer):
        @probe.watch
        def process(x):
            return x * 2

        t = np.ones((4, 8), dtype=np.float32)
        result = process(t)
        assert result.shape == (4, 8)
        out = _capture_to_buffer.getvalue()
        assert "(4, 8)" in out

    def test_watch_no_return(self, _capture_to_buffer):
        @probe.watch(show_return=False)
        def side_effect(x):
            return x + 1

        side_effect(5)
        out = _capture_to_buffer.getvalue()
        assert "called" in out
        assert "done" not in out

    def test_watch_no_args(self, _capture_to_buffer):
        @probe.watch(show_args=False)
        def process(x):
            return x + 1

        process(5)
        out = _capture_to_buffer.getvalue()
        # Should have called but not arg details
        assert "called" in out

    def test_watch_disabled(self, _capture_to_buffer, monkeypatch):
        monkeypatch.setattr(probe_config, "enabled", False)

        @probe.watch
        def add(a, b):
            return a + b

        result = add(3, 4)
        assert result == 7
        assert _capture_to_buffer.getvalue() == ""


# ─── probe.diff ──────────────────────────────────────────────────────────────

class TestDiff:
    def test_identical(self, _capture_to_buffer):
        x = np.ones((4, 4), dtype=np.float32)
        probe.diff(x, x)
        out = _capture_to_buffer.getvalue()
        assert "identical" in out

    def test_different(self, _capture_to_buffer):
        a = np.zeros((4, 4), dtype=np.float32)
        b = np.ones((4, 4), dtype=np.float32)
        probe.diff(a, b)
        out = _capture_to_buffer.getvalue()
        assert "max_abs=" in out
        assert "mean_abs=" in out

    def test_custom_names(self, _capture_to_buffer):
        a = np.zeros((2,), dtype=np.float32)
        b = np.ones((2,), dtype=np.float32)
        probe.diff(a, b, names=("before", "after"))
        out = _capture_to_buffer.getvalue()
        assert "before" in out
        assert "after" in out

    def test_shape_mismatch(self, _capture_to_buffer):
        a = np.zeros((4, 4), dtype=np.float32)
        b = np.zeros((3, 3), dtype=np.float32)
        probe.diff(a, b)
        out = _capture_to_buffer.getvalue()
        assert "shape mismatch" in out

    def test_custom_tag(self, _capture_to_buffer):
        a = np.zeros((2,), dtype=np.float32)
        b = np.ones((2,), dtype=np.float32)
        probe.diff(a, b, tag="ATTN")
        assert "ATTN" in _capture_to_buffer.getvalue()

    def test_close_tensors(self, _capture_to_buffer):
        a = np.ones((4, 4), dtype=np.float64)
        b = a + 1e-7
        probe.diff(a, b)
        out = _capture_to_buffer.getvalue()
        assert "allclose" in out

    def test_disabled(self, _capture_to_buffer, monkeypatch):
        monkeypatch.setattr(probe_config, "enabled", False)
        probe.diff(np.zeros(4), np.ones(4))
        assert _capture_to_buffer.getvalue() == ""