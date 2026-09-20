from contextlib import contextmanager

import torch

from sandbox import anti_hack


_KERNEL_STATE = {"enabled": True}


@contextmanager
def _disabled_kernel():
    previous = _KERNEL_STATE["enabled"]
    _KERNEL_STATE["enabled"] = False
    try:
        yield
    finally:
        _KERNEL_STATE["enabled"] = previous


def test_replay_detects_return_value_fallback(monkeypatch):
    monkeypatch.setattr(anti_hack, "disable_triton_jit", _disabled_kernel)

    def fallback():
        return torch.arange(4)

    is_hack, _ = anti_hack.dual_execution_check(fallback, {})
    assert is_hack


def test_replay_does_not_reuse_mutated_output_buffer(monkeypatch):
    monkeypatch.setattr(anti_hack, "disable_triton_jit", _disabled_kernel)

    def kernel(out):
        if _KERNEL_STATE["enabled"]:
            out.add_(1)

    original = torch.zeros(4)
    is_hack, _ = anti_hack.dual_execution_check(kernel, {"out": original})
    assert not is_hack
    assert torch.equal(original, torch.zeros_like(original))


def test_replay_detects_in_place_framework_fallback(monkeypatch):
    monkeypatch.setattr(anti_hack, "disable_triton_jit", _disabled_kernel)

    def fallback(out):
        out.add_(1)

    is_hack, _ = anti_hack.dual_execution_check(
        fallback, {"out": torch.zeros(4)}
    )
    assert is_hack


def test_replay_restores_rng_state(monkeypatch):
    monkeypatch.setattr(anti_hack, "disable_triton_jit", _disabled_kernel)

    def stochastic_fallback():
        return torch.rand(8)

    is_hack, _ = anti_hack.dual_execution_check(stochastic_fallback, {})
    assert is_hack
