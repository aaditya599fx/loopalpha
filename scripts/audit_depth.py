"""Measure whether occurrence gradients reconstruct ordinary autograd."""

import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import torch

from loopalpha.models import LoopModel


def measure(seed, depth):
    torch.manual_seed(seed)
    x = torch.randn(96, 5, 3, dtype=torch.float64)
    y = torch.randn(96, dtype=torch.float64)
    model = LoopModel((5, 3), 24, depth).double()

    def backward(probe):
        model.zero_grad(set_to_none=True)
        start = perf_counter()
        result = model(x, probe=probe)
        if probe:
            prediction, leaves = result
        else:
            prediction, leaves = result, None
        (prediction - y).square().mean().backward()
        return perf_counter() - start, leaves

    for probe in (False, True):
        backward(probe)
    base_times, probe_times = [], []
    for _ in range(30):
        elapsed, _ = backward(False)
        base_times.append(elapsed)
        expected = torch.cat([p.grad.flatten() for p in model.blocks[0].parameters()])
        elapsed, leaves = backward(True)
        probe_times.append(elapsed)

    grads = torch.stack([torch.cat([p.grad.flatten() for p in state.values()]) for state in leaves])
    norms = torch.linalg.vector_norm(grads, dim=1)
    normalized = grads / norms[:, None].clamp_min(1e-15)
    reconstructed = grads.sum(dim=0)
    torch.testing.assert_close(expected, reconstructed, atol=1e-11, rtol=1e-9)
    return {"seed": seed, "depth": depth, "max_abs_sum_error": (expected - reconstructed).abs().max().item(),
            "gradient_norms": norms.tolist(), "pairwise_cosines": (normalized @ normalized.T).tolist(),
            "tied_forward_backward_ms": 1000 * median(base_times),
            "probe_forward_backward_ms": 1000 * median(probe_times),
            "extra_copy_and_gradient_bytes": 2 * grads.numel() * grads.element_size()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = [measure(seed, depth) for depth in (1, 2, 4, 8) for seed in (17, 41, 73)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
