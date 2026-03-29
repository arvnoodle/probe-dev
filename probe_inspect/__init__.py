"""
probe-inspect — A one-call inspection print for ML research.

Usage:
    from probe_inspect import probe
    from probe_inspect import pb      # short alias — same function!

    x = 42
    pb(x)
    # [script.py:4] x = 42  (int)

    import torch
    t = torch.randn(8, 64, 512)
    pb(t)
    # [script.py:8] t: float32 (8, 64, 512) 262.1K params | μ=-0.001 σ=1.000 ∈[-3.89, 4.02] — cuda:0

    # Decorator — auto-probe inputs/outputs:
    @pb.watch
    def forward(self, x):
        ...

    # Scan all model parameters:
    pb.model_summary(model)

    # Compare two tensors:
    pb.diff(tensor_a, tensor_b)

    # Auto-hook PyTorch model layers:
    handles = pb.hooks(model, layers=["attn"])
"""

from .core import probe, probe_config, ProbeConfig, watch, model_summary, diff, hooks

# Short alias — pb is the same function as probe, with all methods attached
pb = probe

__version__ = "1.1.0"
__all__ = ["probe", "pb", "probe_config", "ProbeConfig", "watch", "model_summary", "diff", "hooks"]