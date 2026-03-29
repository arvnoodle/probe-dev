"""
probe core — inspect anything, one call.

Works on every type:
  - Scalars, strings, bools, None → clean repr with type
  - Lists, dicts, tuples, sets   → pretty-printed with depth control
  - Tensors (torch/numpy/jax/tf) → dtype, shape, param count, stats, device, NaN/Inf warnings

Jupyter-aware: shows cell-local line numbers like [Cell 3 L5].
"""

import inspect
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, TextIO


# ─── ANSI Colors ────────────────────────────────────────────────────────────

class _C:
    """Color palette — muted for long sessions."""
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"

    GRAY    = "\033[90m"
    WHITE   = "\033[97m"
    CYAN    = "\033[36m"
    YELLOW  = "\033[93m"
    GREEN   = "\033[32m"
    MAGENTA = "\033[35m"
    RED     = "\033[91m"
    BLUE    = "\033[94m"

    # Semantic
    LOC      = "\033[90m"
    NAME     = "\033[1;97m"
    SHAPE    = "\033[36m"
    DTYPE    = "\033[33m"
    STAT     = "\033[32m"
    WARN     = "\033[91m"
    VAL      = "\033[93m"
    STR_VAL  = "\033[32m"
    BOOL_VAL = "\033[35m"
    COLL     = "\033[33m"
    SEP      = "\033[90m"
    DEVICE   = "\033[94m"


class _NoColor:
    """Stand-in when colors are off."""
    def __getattr__(self, _):
        return ""

_NO = _NoColor()


# ─── Environment Detection ──────────────────────────────────────────────────

def _in_jupyter() -> bool:
    try:
        from IPython import get_ipython
        shell = get_ipython()
        if shell is None:
            return False
        return shell.__class__.__name__ in (
            "ZMQInteractiveShell",
            "TerminalInteractiveShell",
        )
    except (ImportError, AttributeError):
        return False


def _get_jupyter_cell_info():
    """Return (cell_execution_count, cell_source) or None."""
    try:
        from IPython import get_ipython
        shell = get_ipython()
        if shell is None:
            return None
        exec_count = shell.execution_count
        raw = shell.history_manager.input_hist_raw
        cell_source = raw[-1] if raw else None
        return exec_count, cell_source
    except Exception:
        return None


# ─── Source Introspection ────────────────────────────────────────────────────

def _get_call_info(stack_depth: int = 2):
    """
    Walk the stack to find caller's location and source line.
    In Jupyter: returns cell-local line numbers.
    In scripts:  returns filename:lineno.
    """
    frame = inspect.currentframe()
    try:
        for _ in range(stack_depth):
            if frame is not None:
                frame = frame.f_back
        if frame is None:
            return "?", ""

        info = inspect.getframeinfo(frame)
        abs_lineno = info.lineno
        func = info.function
        source_line = info.code_context[0].strip() if info.code_context else ""

        # ── Jupyter ──
        jupyter_info = _get_jupyter_cell_info()
        if jupyter_info is not None:
            cell_num, _ = jupyter_info
            func_str = ""
            if func not in ("<module>",) and not func.startswith("<cell line"):
                func_str = f" | {func}()"
            return f"Cell {cell_num} L{abs_lineno}{func_str}", source_line

        # ── Script ──
        filename = os.path.basename(info.filename)
        func_str = ""
        if func != "<module>":
            func_str = f" | {func}()"
        return f"{filename}:{abs_lineno}{func_str}", source_line

    finally:
        del frame


def _extract_arg_names(source_line: str) -> list[str]:
    """Parse `probe(x, model.weight, 2+2)` → ['x', 'model.weight', '2+2']."""
    match = re.search(r'\bprobe\s*\(', source_line)
    if not match:
        return []

    start = match.end()
    depth = 1
    args: list[list[str]] = [[]]
    i = start

    while i < len(source_line) and depth > 0:
        ch = source_line[i]
        if ch in "([{":
            depth += 1
            args[-1].append(ch)
        elif ch in ")]}":
            depth -= 1
            if depth == 0:
                break
            args[-1].append(ch)
        elif ch == "," and depth == 1:
            args.append([])
        elif ch in ('"', "'"):
            quote = ch
            args[-1].append(ch)
            i += 1
            while i < len(source_line) and source_line[i] != quote:
                if source_line[i] == "\\":
                    args[-1].append(source_line[i])
                    i += 1
                if i < len(source_line):
                    args[-1].append(source_line[i])
                i += 1
            if i < len(source_line):
                args[-1].append(source_line[i])
        else:
            args[-1].append(ch)
        i += 1

    raw = ["".join(a).strip() for a in args]

    # Filter out probe's own kwargs
    _probe_kwargs = {
        "tag", "show_time", "color", "compact", "show_type", "show_name",
        "show_loc", "show_stats", "show_device", "stat_fmt", "prefix", "output",
    }
    return [
        a for a in raw
        if a and not (
            (m := re.match(r"^(\w+)\s*=", a)) and m.group(1) in _probe_kwargs
        )
    ]


# ─── Tensor Detection & Summary ─────────────────────────────────────────────

def _is_tensor(val: Any) -> bool:
    """Check for tensor/ndarray from any major framework via duck typing."""
    return (
        hasattr(val, "shape")
        and hasattr(val, "dtype")
        and not isinstance(val, type)
    )


def _tensor_summary(val: Any, c, show_stats: bool, show_device: bool, stat_fmt: str) -> str:
    """One-line tensor summary: dtype shape params | stats — device."""
    parts = []

    # dtype
    dtype_str = str(val.dtype).replace("torch.", "").replace("numpy.", "").replace("tf.", "")
    parts.append(f"{c.DTYPE}{dtype_str}{c.RESET}")

    # shape
    shape = tuple(val.shape)
    shape_str = f"({', '.join(str(d) for d in shape)})" if shape else "(scalar)"
    parts.append(f"{c.SHAPE}{shape_str}{c.RESET}")

    # numel
    numel = 1
    for d in shape:
        numel *= d
    if numel > 1:
        if numel >= 1_000_000_000:
            n = f"{numel / 1e9:.2f}B"
        elif numel >= 1_000_000:
            n = f"{numel / 1e6:.2f}M"
        elif numel >= 1_000:
            n = f"{numel / 1e3:.1f}K"
        else:
            n = str(numel)
        parts.append(f"{c.DIM}{n} params{c.RESET}")

    # stats
    if show_stats and numel > 0:
        stat_str = _compute_stats(val, stat_fmt, c)
        if stat_str:
            parts.append(f"{c.SEP}|{c.RESET} {stat_str}")

    # device
    if show_device and hasattr(val, "device"):
        dev = str(val.device)
        if dev and dev != "cpu":
            parts.append(f"{c.SEP}—{c.RESET} {c.DEVICE}{dev}{c.RESET}")
        elif dev == "cpu":
            parts.append(f"{c.SEP}—{c.RESET} {c.DIM}cpu{c.RESET}")

    return " ".join(parts)


def _compute_stats(val: Any, fmt: str, c) -> Optional[str]:
    """Compute μ, σ, min, max and flag NaN/Inf."""
    try:
        import numpy as np

        v = val
        if hasattr(v, "detach"):
            v = v.detach()
        if hasattr(v, "cpu"):
            v = v.cpu()
        if hasattr(v, "numpy"):
            try:
                v = v.numpy()
            except Exception:
                pass
        elif hasattr(v, "asnumpy"):
            v = v.asnumpy()

        arr = np.asarray(v)
        if arr.size == 0:
            return None

        # Cast to float for stats
        if not np.issubdtype(arr.dtype, np.floating):
            arr = arr.astype(np.float64)

        nan_count = int(np.isnan(arr).sum())
        inf_count = int(np.isinf(arr).sum())
        finite = arr[np.isfinite(arr)]

        if finite.size == 0:
            return f"{c.WARN}all NaN/Inf!{c.RESET}"

        mean = float(finite.mean())
        std = float(finite.std())
        vmin = float(finite.min())
        vmax = float(finite.max())

        s = f"{c.STAT}μ={mean:{fmt}} σ={std:{fmt}} ∈[{vmin:{fmt}}, {vmax:{fmt}}]{c.RESET}"
        warnings = []
        if nan_count > 0:
            warnings.append(f"{c.WARN}{nan_count} NaN{c.RESET}")
        if inf_count > 0:
            warnings.append(f"{c.WARN}{inf_count} Inf{c.RESET}")
        if warnings:
            s += " " + " ".join(warnings)
        return s

    except Exception:
        return None


# ─── Value Formatting (non-tensor) ──────────────────────────────────────────

def _format_value(
    val: Any, c, compact: bool,
    max_depth: int = 3, max_items: int = 20, depth: int = 0,
) -> str:
    """Pretty-format any non-tensor value."""
    if depth > max_depth:
        return f"{c.DIM}...{c.RESET}"

    if val is None:
        return f"{c.BOOL_VAL}None{c.RESET}"
    if isinstance(val, bool):
        return f"{c.BOOL_VAL}{val}{c.RESET}"
    if isinstance(val, (int, float, complex)):
        return f"{c.VAL}{val}{c.RESET}"
    if isinstance(val, str):
        preview = (val[:200] + "...") if len(val) > 200 else val
        return f"{c.STR_VAL}'{preview}'{c.RESET}"
    if isinstance(val, bytes):
        return f"{c.STR_VAL}{val!r}{c.RESET}"

    if compact:
        r = repr(val)
        return f"{c.VAL}{r[:117] + '...' if len(r) > 120 else r}{c.RESET}"

    indent = "  " * (depth + 1)
    close = "  " * depth

    if isinstance(val, dict):
        if not val:
            return f"{c.COLL}{{}}{c.RESET}"
        items = list(val.items())[:max_items]
        parts = [
            f"{indent}{_format_value(k, c, False, max_depth, max_items, depth+1)}"
            f"{c.SEP}: {c.RESET}"
            f"{_format_value(v, c, False, max_depth, max_items, depth+1)}"
            for k, v in items
        ]
        extra = f"\n{indent}{c.DIM}... +{len(val) - max_items} more{c.RESET}" if len(val) > max_items else ""
        return f"{c.COLL}{{{c.RESET}\n" + ",\n".join(parts) + extra + f"\n{close}{c.COLL}}}{c.RESET}"

    if isinstance(val, (list, tuple, set, frozenset)):
        if not val:
            br = "[]" if isinstance(val, list) else ("()" if isinstance(val, tuple) else "{}")
            return f"{c.COLL}{br}{c.RESET}"
        items = list(val)[:max_items]
        parts = [
            f"{indent}{_format_value(v, c, False, max_depth, max_items, depth+1)}"
            for v in items
        ]
        extra = f"\n{indent}{c.DIM}... +{len(val) - max_items} more{c.RESET}" if len(val) > max_items else ""
        ob, cb = {list: ("[", "]"), tuple: ("(", ")"), set: ("{", "}"), frozenset: ("{", "}")}.get(type(val), ("[", "]"))
        return f"{c.COLL}{ob}{c.RESET}\n" + ",\n".join(parts) + extra + f"\n{close}{c.COLL}{cb}{c.RESET}"

    r = repr(val)
    if len(r) > 300:
        r = r[:297] + "..."
    return f"{c.VAL}{r}{c.RESET}"


# ─── Config ─────────────────────────────────────────────────────────────────

@dataclass
class ProbeConfig:
    """Global settings for probe. Modify the singleton `probe_config`."""

    enabled: bool = True
    color: bool = True
    show_name: bool = True
    show_type: bool = True        # for non-tensors: (int), (str)
    show_loc: bool = True
    show_stats: bool = True       # tensor μ, σ, min, max
    show_device: bool = True      # tensor device
    show_time: bool = False
    compact: bool = False
    stat_fmt: str = ".4f"
    time_fmt: str = "%H:%M:%S.%f"
    output: TextIO = field(default_factory=lambda: sys.stderr)
    prefix: str = ""
    tag: str = ""


# Singleton
probe_config = ProbeConfig()


# ─── Main Function ──────────────────────────────────────────────────────────

def probe(
    *values: Any,
    tag: Optional[str] = None,
    show_time: Optional[bool] = None,
    color: Optional[bool] = None,
    compact: Optional[bool] = None,
    show_type: Optional[bool] = None,
    show_name: Optional[bool] = None,
    show_loc: Optional[bool] = None,
    show_stats: Optional[bool] = None,
    show_device: Optional[bool] = None,
    stat_fmt: Optional[str] = None,
    prefix: Optional[str] = None,
    output: Optional[TextIO] = None,
    _stack_depth: int = 2,
) -> Any:
    """
    Inspect any value — scalars, strings, dicts, tensors, anything.

    Tensors get shape/dtype/stats/device.
    Everything else gets a clean repr with type.
    Returns the value for inline use: `result = probe(model(x))`.

    Args:
        *values:      One or more values to inspect.
        tag:          Label for filtering output, e.g. "ATTN", "GRAD".
        show_time:    Prepend timestamp.
        color:        ANSI colors on/off.
        compact:      Force single-line repr for collections.
        show_type:    Show (int), (str) for non-tensors.
        show_name:    Show variable name extracted from source.
        show_loc:     Show [file:line | func()] location.
        show_stats:   Show μ, σ, min, max for tensors.
        show_device:  Show tensor device.
        stat_fmt:     Float format for stats (default ".4f").
        prefix:       Custom string prefix.
        output:       Output stream (default stderr).

    Returns:
        The original value (single arg) or tuple (multiple args).
    """
    cfg = probe_config
    if not cfg.enabled:
        return values[0] if len(values) == 1 else values

    # Resolve per-call overrides
    _color     = color if color is not None else cfg.color
    _show_time = show_time if show_time is not None else cfg.show_time
    _compact   = compact if compact is not None else cfg.compact
    _show_type = show_type if show_type is not None else cfg.show_type
    _show_name = show_name if show_name is not None else cfg.show_name
    _show_loc  = show_loc if show_loc is not None else cfg.show_loc
    _show_stat = show_stats if show_stats is not None else cfg.show_stats
    _show_dev  = show_device if show_device is not None else cfg.show_device
    _stat_fmt  = stat_fmt if stat_fmt is not None else cfg.stat_fmt
    _prefix    = prefix if prefix is not None else cfg.prefix
    _tag       = tag if tag is not None else cfg.tag
    _output    = output if output is not None else cfg.output

    c = _C() if _color else _NO

    # Introspect call site
    location, source_line = _get_call_info(_stack_depth)
    arg_names = _extract_arg_names(source_line)

    # ── Location bracket ──
    loc_parts = []
    if _show_loc:
        loc_parts.append(location)
    if _tag:
        loc_parts.append(_tag)

    loc_str = ""
    if loc_parts:
        inner = " | ".join(loc_parts)
        loc_str = f"{c.LOC}[{inner}]{c.RESET}"

    # ── Timestamp ──
    time_str = ""
    if _show_time:
        now = datetime.now().strftime(cfg.time_fmt)
        if cfg.time_fmt.endswith("%f"):
            now = now[:-3]
        time_str = f"{c.DIM}{now}{c.RESET} "

    # ── Prefix ──
    pfx = f"{_prefix} " if _prefix else ""

    # ── Format each value ──
    entries = []
    for i, val in enumerate(values):
        name = arg_names[i] if i < len(arg_names) else None

        if _is_tensor(val):
            summary = _tensor_summary(val, c, _show_stat, _show_dev, _stat_fmt)
            if _show_name and name:
                entry = f"{c.NAME}{name}{c.RESET}{c.SEP}:{c.RESET} {summary}"
            else:
                entry = summary
        else:
            val_str = _format_value(val, c, _compact)
            type_str = f"  {c.DIM}({type(val).__name__}){c.RESET}" if _show_type else ""
            if _show_name and name and not name.startswith("arg"):
                entry = f"{c.NAME}{name}{c.RESET} {c.SEP}={c.RESET} {val_str}{type_str}"
            else:
                entry = f"{val_str}{type_str}"

        entries.append(entry)

    # ── Assemble ──
    if len(entries) == 1:
        body = entries[0]
    elif _compact:
        body = f"{c.SEP}, {c.RESET}".join(entries)
    else:
        body = "\n" + "\n".join(f"  {e}" for e in entries)

    line = f"{time_str}{pfx}{loc_str} {body}" if loc_str else f"{time_str}{pfx}{body}"

    print(line.strip(), file=_output)

    return values[0] if len(values) == 1 else values


# ─── Decorator: @probe.watch ────────────────────────────────────────────────

def watch(fn=None, *, tag: Optional[str] = None, show_args: bool = True, show_return: bool = True):
    """
    Decorator — auto-probe function inputs and outputs.

    Usage:
        @probe.watch
        def forward(self, x, mask=None):
            ...

        @probe.watch(tag="ATTN", show_args=False)
        def attention(Q, K, V):
            ...
    """
    import functools

    def decorator(func):
        _tag = tag or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cfg = probe_config
            if not cfg.enabled:
                return func(*args, **kwargs)

            _output = cfg.output
            c = _C() if cfg.color else _NO

            # Input summary
            sig_parts = []
            params = list(inspect.signature(func).parameters.keys())
            if show_args:
                for i, a in enumerate(args):
                    name = params[i] if i < len(params) else f"arg{i}"
                    if name == "self":
                        continue
                    if _is_tensor(a):
                        dtype_s = str(a.dtype).replace("torch.", "").replace("numpy.", "")
                        shape_s = "(" + ", ".join(str(d) for d in a.shape) + ")"
                        sig_parts.append(f"{c.NAME}{name}{c.RESET}{c.SEP}:{c.RESET} {c.DTYPE}{dtype_s}{c.RESET} {c.SHAPE}{shape_s}{c.RESET}")
                    else:
                        sig_parts.append(f"{c.NAME}{name}{c.RESET}{c.SEP}={c.RESET}{c.VAL}{repr(a)[:60]}{c.RESET}")

            header = f"{c.LOC}[{_tag}]{c.RESET} {c.DIM}→ called{c.RESET}"
            if sig_parts:
                header += f"  {c.SEP}({c.RESET}{f'{c.SEP}, {c.RESET}'.join(sig_parts)}{c.SEP}){c.RESET}"
            print(header, file=_output)

            result = func(*args, **kwargs)

            if show_return:
                if _is_tensor(result):
                    summary = _tensor_summary(result, c, cfg.show_stats, cfg.show_device, cfg.stat_fmt)
                    ret_str = f"{c.NAME}return{c.RESET}{c.SEP}:{c.RESET} {summary}"
                elif isinstance(result, tuple) and any(_is_tensor(r) for r in result):
                    parts = []
                    for j, r in enumerate(result):
                        if _is_tensor(r):
                            s = _tensor_summary(r, c, cfg.show_stats, cfg.show_device, cfg.stat_fmt)
                            parts.append(f"[{j}]{c.SEP}:{c.RESET} {s}")
                        else:
                            parts.append(f"[{j}]{c.SEP}={c.RESET}{c.VAL}{repr(r)[:60]}{c.RESET}")
                    ret_str = f"{c.NAME}return{c.RESET}\n" + "\n".join(f"  {p}" for p in parts)
                else:
                    ret_str = f"{c.NAME}return{c.RESET} {c.SEP}={c.RESET} {c.VAL}{repr(result)[:120]}{c.RESET}  {c.DIM}({type(result).__name__}){c.RESET}"

                print(f"{c.LOC}[{_tag}]{c.RESET} {c.DIM}← done{c.RESET}   {ret_str}", file=_output)

            return result
        return wrapper

    if fn is not None:
        return decorator(fn)
    return decorator


# ─── Model Parameter Scanner ────────────────────────────────────────────────

def model_summary(model, tag: str = "MODEL") -> None:
    """
    Probe all parameters in a PyTorch nn.Module.

    Usage:
        probe.model_summary(model)
        # [MODEL] transformer.wte.weight         : float32 (50257, 768)  38.60M params
        # [MODEL] transformer.wpe.weight         : float32 (1024, 768)   786.4K params
        # ...
        # [MODEL] Total: 124.44M parameters (473.89 MB)
    """
    cfg = probe_config
    if not cfg.enabled:
        return

    c = _C() if cfg.color else _NO
    _output = cfg.output

    total_params = 0
    trainable_params = 0
    max_name_len = 0

    param_list = []
    try:
        for name, param in model.named_parameters():
            param_list.append((name, param))
            max_name_len = max(max_name_len, len(name))
    except AttributeError:
        print(f"{c.LOC}[{tag}]{c.RESET} {c.WARN}Not a PyTorch module (no named_parameters){c.RESET}", file=_output)
        return

    if not param_list:
        print(f"{c.LOC}[{tag}]{c.RESET} {c.DIM}No parameters found{c.RESET}", file=_output)
        return

    for name, param in param_list:
        numel = param.numel()
        total_params += numel
        if param.requires_grad:
            trainable_params += numel

        dtype_s = str(param.dtype).replace("torch.", "")
        shape_s = "(" + ", ".join(str(d) for d in param.shape) + ")"

        if numel >= 1_000_000:
            n_str = f"{numel / 1e6:.2f}M"
        elif numel >= 1_000:
            n_str = f"{numel / 1e3:.1f}K"
        else:
            n_str = str(numel)

        padded_name = name.ljust(max_name_len)
        frozen = "" if param.requires_grad else f" {c.DIM}(frozen){c.RESET}"

        stat_str = ""
        if cfg.show_stats:
            stat_result = _compute_stats(param, cfg.stat_fmt, c)
            if stat_result:
                stat_str = f" {c.SEP}|{c.RESET} {stat_result}"

        line = (
            f"{c.LOC}[{tag}]{c.RESET} "
            f"{c.NAME}{padded_name}{c.RESET} {c.SEP}:{c.RESET} "
            f"{c.DTYPE}{dtype_s}{c.RESET} {c.SHAPE}{shape_s}{c.RESET} "
            f"{c.DIM}{n_str:>8s} params{c.RESET}"
            f"{stat_str}{frozen}"
        )
        print(line, file=_output)

    # Summary
    if total_params >= 1_000_000_000:
        total_str = f"{total_params / 1e9:.2f}B"
    elif total_params >= 1_000_000:
        total_str = f"{total_params / 1e6:.2f}M"
    else:
        total_str = f"{total_params:,}"

    try:
        bytes_per = param_list[0][1].element_size()
    except Exception:
        bytes_per = 4

    total_bytes = total_params * bytes_per
    mem_str = f"{total_bytes / 1e9:.2f} GB" if total_bytes >= 1e9 else f"{total_bytes / 1e6:.2f} MB"

    frozen_count = total_params - trainable_params
    frozen_str = f"  {c.DIM}({frozen_count:,} frozen){c.RESET}" if frozen_count > 0 else ""

    print(
        f"{c.LOC}[{tag}]{c.RESET} {c.BOLD}Total: {total_str} parameters ({mem_str}){c.RESET}{frozen_str}",
        file=_output,
    )


# ─── Tensor Diff ─────────────────────────────────────────────────────────────

def diff(a, b, names: Optional[tuple] = None, tag: str = "DIFF") -> None:
    """
    Compare two tensors — shapes, stats, max absolute/relative difference.

    Usage:
        probe.diff(tensor_before, tensor_after)
        probe.diff(my_attn, reference_attn, names=("mine", "ref"))
    """
    cfg = probe_config
    if not cfg.enabled:
        return

    c = _C() if cfg.color else _NO
    _output = cfg.output
    fmt = cfg.stat_fmt

    name_a = names[0] if names else "a"
    name_b = names[1] if names else "b"

    if _is_tensor(a):
        sa = _tensor_summary(a, c, cfg.show_stats, cfg.show_device, fmt)
        print(f"{c.LOC}[{tag}]{c.RESET} {c.NAME}{name_a}{c.RESET}{c.SEP}:{c.RESET} {sa}", file=_output)
    else:
        print(f"{c.LOC}[{tag}]{c.RESET} {c.NAME}{name_a}{c.RESET} {c.SEP}={c.RESET} {c.VAL}{repr(a)[:120]}{c.RESET}", file=_output)

    if _is_tensor(b):
        sb = _tensor_summary(b, c, cfg.show_stats, cfg.show_device, fmt)
        print(f"{c.LOC}[{tag}]{c.RESET} {c.NAME}{name_b}{c.RESET}{c.SEP}:{c.RESET} {sb}", file=_output)
    else:
        print(f"{c.LOC}[{tag}]{c.RESET} {c.NAME}{name_b}{c.RESET} {c.SEP}={c.RESET} {c.VAL}{repr(b)[:120]}{c.RESET}", file=_output)

    if _is_tensor(a) and _is_tensor(b):
        try:
            import numpy as np
            arr_a = _to_numpy(a)
            arr_b = _to_numpy(b)

            if arr_a.shape != arr_b.shape:
                print(f"{c.LOC}[{tag}]{c.RESET} {c.WARN}shape mismatch: {arr_a.shape} vs {arr_b.shape}{c.RESET}", file=_output)
                return

            delta = arr_a.astype(np.float64) - arr_b.astype(np.float64)
            abs_delta = np.abs(delta)
            max_abs = float(abs_delta.max())
            mean_abs = float(abs_delta.mean())

            denom = np.maximum(np.abs(arr_b.astype(np.float64)), 1e-12)
            max_rel = float((abs_delta / denom).max())
            mean_rel = float((abs_delta / denom).mean())

            if max_abs == 0.0:
                print(f"{c.LOC}[{tag}]{c.RESET} {c.STAT}identical ✓{c.RESET}", file=_output)
            else:
                allclose = np.allclose(arr_a, arr_b, atol=1e-6, rtol=1e-5)
                close_str = f"  {c.STAT}(≈ allclose){c.RESET}" if allclose else ""
                print(
                    f"{c.LOC}[{tag}]{c.RESET} {c.NAME}Δ{c.RESET}{c.SEP}:{c.RESET} "
                    f"{c.STAT}max_abs={max_abs:{fmt}} mean_abs={mean_abs:{fmt}} "
                    f"max_rel={max_rel:{fmt}} mean_rel={mean_rel:{fmt}}{c.RESET}{close_str}",
                    file=_output,
                )
        except Exception as e:
            print(f"{c.LOC}[{tag}]{c.RESET} {c.WARN}diff failed: {e}{c.RESET}", file=_output)


def _to_numpy(val):
    """Convert any tensor to numpy."""
    import numpy as np
    v = val
    if hasattr(v, "detach"):
        v = v.detach()
    if hasattr(v, "cpu"):
        v = v.cpu()
    if hasattr(v, "numpy"):
        try:
            return v.numpy()
        except Exception:
            pass
    if hasattr(v, "asnumpy"):
        return v.asnumpy()
    return np.asarray(v)


# ─── PyTorch Forward Hooks ──────────────────────────────────────────────────

def hooks(model, layers: Optional[list] = None, tag: str = "HOOK") -> list:
    """
    Register forward hooks on a PyTorch model to auto-probe layer outputs.

    Usage:
        handles = probe.hooks(model)                           # all layers
        handles = probe.hooks(model, layers=["attn", "mlp"])   # filter by name
        model(x)  # each hooked layer prints its output
        for h in handles: h.remove()                           # clean up

    Returns:
        List of hook handles.
    """
    cfg = probe_config
    handles = []

    try:
        named_modules = list(model.named_modules())
    except AttributeError:
        c = _C() if cfg.color else _NO
        print(f"{c.LOC}[{tag}]{c.RESET} {c.WARN}Not a PyTorch module{c.RESET}", file=cfg.output)
        return handles

    for name, module in named_modules:
        if not name:
            continue
        if layers is not None:
            if not any(f in name for f in layers):
                continue

        def _make_hook(layer_name):
            def hook_fn(mod, inp, output):
                if not cfg.enabled:
                    return
                c = _C() if cfg.color else _NO
                if _is_tensor(output):
                    s = _tensor_summary(output, c, cfg.show_stats, cfg.show_device, cfg.stat_fmt)
                    print(f"{c.LOC}[{tag} | {layer_name}]{c.RESET} {c.NAME}output{c.RESET}{c.SEP}:{c.RESET} {s}", file=cfg.output)
                elif isinstance(output, tuple):
                    for j, o in enumerate(output):
                        if _is_tensor(o):
                            s = _tensor_summary(o, c, cfg.show_stats, cfg.show_device, cfg.stat_fmt)
                            print(f"{c.LOC}[{tag} | {layer_name}]{c.RESET} {c.NAME}out[{j}]{c.RESET}{c.SEP}:{c.RESET} {s}", file=cfg.output)
            return hook_fn

        h = module.register_forward_hook(_make_hook(name))
        handles.append(h)

    c = _C() if cfg.color else _NO
    print(f"{c.LOC}[{tag}]{c.RESET} {c.DIM}Registered {len(handles)} hooks{c.RESET}", file=cfg.output)
    return handles


# ─── Attach utilities as methods on the probe function ───────────────────────

probe.watch = watch
probe.model_summary = model_summary
probe.diff = diff
probe.hooks = hooks
