"""
probe - A one-call inspection print for ML research.

Usage:
    from probe import probe

    x = 42
    probe(x)
    # [script.py:4] x = 42  (int)

    import torch
    t = torch.randn(8, 64, 512)
    probe(t)
    # [script.py:8] t: float32 (8, 64, 512) 262.1K params | μ=-0.001 σ=1.000 ∈[-3.89, 4.02] — cuda:0

    # Decorator — auto-probe inputs/outputs:
    @probe.watch
    def forward(self, x):
        ...

    # Scan all model parameters:
    probe.model_summary(model)

    # Compare two tensors:
    probe.diff(tensor_a, tensor_b)

    # Auto-hook PyTorch model layers:
    handles = probe.hooks(model, layers=["attn"])
"""

from .core import probe, probe_config, ProbeConfig, watch, model_summary, diff, hooks

__version__ = "1.0.0"
__all__ = ["probe", "probe_config", "ProbeConfig", "watch", "model_summary", "diff", "hooks"]
