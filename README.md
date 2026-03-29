# probe

A one-call inspection print for ML research. See tensor shapes, dtypes, stats, NaN warnings, and your exact location — without writing a single f-string.

Built for researchers who just want to check what's happening at line 3432 of their transformer.

## Install

```bash
pip install probe-inspect
```

## Quick Start

```python
from probe import probe

# Works on everything
x = 42
probe(x)
# [train.py:4] x = 42  (int)

name = "gpt-nano"
probe(name)
# [train.py:7] name = 'gpt-nano'  (str)

config = {"d_model": 512, "n_heads": 8, "layers": 12}
probe(config)
# [train.py:10] config = {
#   'd_model': 512,
#   'n_heads': 8,
#   'layers': 12
# }  (dict)

# Tensors get the full treatment
import torch
hidden = torch.randn(8, 128, 512, device="cuda")
probe(hidden)
# [train.py:18] hidden: float32 (8, 128, 512) 524.3K params | μ=-0.001 σ=1.002 ∈[-4.10, 3.89] — cuda:0
```

## Works in Jupyter

Line numbers are **cell-local** — `[Cell 3 L5]` means Cell 3, Line 5. Re-run a cell, numbers still make sense.

```python
# Cell [3]:
embeddings = model.embed(tokens)     # L1
probe(embeddings)                     # L2
# [Cell 3 L2] embeddings: float32 (32, 128, 768) 3.15M params | μ=0.002 σ=0.881 ∈[-3.41, 3.52] — cuda:0
```

Set output to stdout so it shows in cell output:
```python
import sys
from probe import probe, probe_config
probe_config.output = sys.stdout
```

## Features

### Any value, any type
```python
probe(42)                  # [file:1] x = 42  (int)
probe("hello")             # [file:2] name = 'hello'  (str)
probe(True)                # [file:3] flag = True  (bool)
probe([1, 2, 3])           # [file:4] dims = [1, 2, 3]  (list)
probe(my_custom_object)    # [file:5] obj = MyClass(...)  (MyClass)
```

### Tensor inspection (torch, numpy, jax, tensorflow)
```python
probe(attention_weights)
# [model.py:42 | forward()] attention_weights: float32 (8, 12, 128, 128) 1.57M params | μ=0.008 σ=0.029 ∈[0.000, 1.000] — cuda:0
```

### NaN / Inf warnings — lights up red
```python
probe(gradient)
# [model.py:87 | backward()] gradient: float32 (512, 512) 262.1K params | μ=0.03 σ=1.45 ∈[-8.2, 7.1] 3 NaN 1 Inf
```

### Multiple values
```python
probe(Q, K, V)
# [model.py:35 | attention()]
#   Q: float32 (8, 64, 512) ...
#   K: float32 (8, 64, 512) ...
#   V: float32 (8, 64, 512) ...
```

### Tags for filtering
```python
probe(attn_scores, tag="ATTN")    # [model.py:42 | ATTN] attn_scores: ...
probe(gradient, tag="GRAD")        # [model.py:87 | GRAD] gradient: ...
```

### Pass-through return
```python
# probe returns the value — no need to restructure your code
output = probe(model(x))
loss = probe(criterion(output, labels))
```

### Function context — automatic
```python
def scaled_dot_product(Q, K, V):
    probe(Q)
    # [model.py:42 | scaled_dot_product()] Q: float32 (8, 64, 512) ...
```

### Timestamps
```python
probe(loss, show_time=True)
# 14:32:01.234 [train.py:89] loss = 0.4523  (float)
```

### `@probe.watch` — auto-probe function I/O
```python
@probe.watch
def scaled_dot_product(Q, K, V):
    scores = Q @ K.T / math.sqrt(Q.shape[-1])
    return softmax(scores) @ V

output = scaled_dot_product(Q, K, V)
# [scaled_dot_product] → called  (Q: float32 (64, 512), K: float32 (64, 512), V: float32 (64, 512))
# [scaled_dot_product] ← done   return: float32 (64, 512) | μ=0.001 σ=0.312 ∈[-1.27, 1.53]

# With options:
@probe.watch(tag="ATTN", show_return=False)
def attention(Q, K, V):
    ...
```

### `probe.model_summary()` — scan all parameters
```python
probe.model_summary(model)
# [MODEL] transformer.wte.weight         : float32 (50257, 768)  38.60M params | μ=-0.0002 σ=0.0200
# [MODEL] transformer.wpe.weight         : float32 (1024, 768)   786.4K params | μ=0.0001 σ=0.0100
# [MODEL] transformer.ln_f.weight        : float32 (768,)           768 params | μ=1.0000 σ=0.0000
# ...
# [MODEL] Total: 124.44M parameters (473.89 MB)
```

### `probe.diff()` — compare two tensors
```python
probe.diff(my_output, reference_output, names=("mine", "ref"))
# [DIFF] mine: float32 (8, 128, 512) | μ=0.002 σ=0.998 ...
# [DIFF] ref:  float32 (8, 128, 512) | μ=0.001 σ=1.001 ...
# [DIFF] Δ: max_abs=0.0312 mean_abs=0.0001 max_rel=0.0524 mean_rel=0.0002  (≈ allclose)
```

### `probe.hooks()` — auto-probe PyTorch layer outputs
```python
handles = probe.hooks(model, layers=["attn", "mlp"])
output = model(x)
# [HOOK | encoder.layer.0.attn] output: float32 (8, 128, 512) | μ=...
# [HOOK | encoder.layer.0.mlp]  output: float32 (8, 128, 512) | μ=...
# ...

for h in handles: h.remove()  # clean up
```

## Global Config

```python
from probe import probe_config

probe_config.show_time = True       # Always show timestamps
probe_config.color = False           # No ANSI colors (for log files)
probe_config.show_stats = False      # Hide μ, σ, min, max
probe_config.stat_fmt = ".2f"        # Fewer decimal places
probe_config.compact = True          # Force single-line output
probe_config.enabled = False         # Silence everything
probe_config.tag = "LAYER3"          # Persistent tag on all output
probe_config.output = open("run.log", "a")  # Log to file
```

## All Options

| Option         | Default       | Description                            |
|----------------|---------------|----------------------------------------|
| `enabled`      | `True`        | Master on/off switch                   |
| `color`        | `True`        | ANSI terminal colors                   |
| `show_name`    | `True`        | Variable / expression name             |
| `show_type`    | `True`        | Type for non-tensors: `(int)`, `(str)` |
| `show_loc`     | `True`        | `[file.py:42 \| func()]`              |
| `show_stats`   | `True`        | Tensor μ, σ, min, max                 |
| `show_device`  | `True`        | Tensor device (cuda:0, cpu)            |
| `show_time`    | `False`       | Timestamp prefix                       |
| `compact`      | `False`       | Force single-line for collections      |
| `stat_fmt`     | `".4f"`       | Float format for tensor stats          |
| `time_fmt`     | `%H:%M:%S.%f` | Timestamp format                      |
| `output`       | `sys.stderr`  | Output stream                          |
| `tag`          | `""`          | Persistent tag on all output           |

## License

MIT
