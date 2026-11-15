"""Measure method-local CPU process RSS and sampled registered-operator FLOPs."""

import argparse
import json
import platform
from pathlib import Path
import resource
import subprocess
import sys

import torch
import yaml

from extended_benchmark import task_data
from loopalpha.experiment import OPTIMIZERS
from loopalpha.training import fit


def worker(method):
    torch.set_num_threads(1)
    config = yaml.safe_load(Path("configs/research.yaml").read_text())
    samples = task_data("nonlinear_recurrence", config)
    train, val = samples.split(0, 90), samples.split(90, 120)
    rate = config["grids"][method][1]
    _, _, _, run, _ = fit(samples, train, val, method, rate, 17,
                           {**config, "steps": 100, "eval_every": 20})
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True) as prof:
        fit(samples, train, val, method, rate, 17, {**config, "steps": 20, "eval_every": 20})
    flops = sum(event.flops for event in prof.key_averages())
    return {"method": method, "peak_rss_bytes": int(rss if platform.system() == "Darwin" else rss * 1024),
            "profile_total_registered_flops": flops, "profile_steps": 20,
            "timed_steps": 100, "timed_seconds": run["seconds"],
            "probe_parameter_bytes": run["probe_parameter_bytes"], "profile_failed": run["failed"]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path)
    p.add_argument("--worker", choices=OPTIMIZERS)
    args = p.parse_args()
    if args.worker:
        print(json.dumps(worker(args.worker), allow_nan=False))
        return
    if not args.out:
        p.error("--out is required")
    rows = []
    for method in OPTIMIZERS:
        payload = subprocess.check_output([sys.executable, __file__, "--worker", method], text=True)
        rows.append(json.loads(payload))
    with args.out.open("x") as handle:
        json.dump(rows, handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
