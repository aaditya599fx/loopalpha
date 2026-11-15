"""Compare optimizers at common observed wall times and update counts."""

import argparse
import hashlib
import json
import platform
import resource
import subprocess
from pathlib import Path
from statistics import mean

import numpy as np
import torch
import yaml

from loopalpha.data import toy_samples
from loopalpha.experiment import OPTIMIZERS, predict
from loopalpha.training import fit


SEEDS = [17, 41, 73, 101, 137, 173, 211, 251, 293, 337]
STEP_BUDGETS = [100, 300, 600, 1000]
TIME_BUDGETS = [.12, .30, .60, 1.20]


def task_data(task, config):
    samples = toy_samples(config["data_seed"], 150, 16, 5)
    if task == "delayed_interaction":
        x = samples.x
        samples.y = (.7 * np.tanh(x[:, 0, 0] * x[:, 4, 1])
                     + .35 * x[:, 1, 2] - .2 * x[:, 3, 0]).astype(np.float32)
    return samples


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    config_path = Path("configs/research.yaml")
    config = yaml.safe_load(config_path.read_text())
    config.update(steps=STEP_BUDGETS[-1], eval_every=20, time_budget=TIME_BUDGETS[-1])
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    try:
        cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        cpu = platform.processor()
    metadata = {"platform": platform.platform(), "cpu": cpu, "device": "cpu",
                "cuda_available": torch.cuda.is_available(), "threads": 1,
                "torch": torch.__version__, "python": platform.python_version(),
                "numpy": np.__version__, "training_seeds": SEEDS,
                "tuning_seeds": [3, 7], "step_budgets": STEP_BUDGETS, "time_budgets": TIME_BUDGETS,
                "config": config, "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in [Path(__file__).resolve(), *Path("src/loopalpha").glob("*.py")]}
                }
    metadata["source_sha256"][Path(__file__).name] = metadata["source_sha256"].pop(str(Path(__file__).resolve()))
    args.out.mkdir(parents=True)
    for task in ["nonlinear_recurrence", "delayed_interaction"]:
        print(f"Benchmarking {task}", flush=True)
        samples = task_data(task, config)
        train, val, test = (samples.split(a, b) for a, b in [(0, 90), (90, 120), (120, 150)])
        tuning = []
        selected = {}
        runs = []
        for method in OPTIMIZERS:
            candidates = []
            for rate in config["grids"][method]:
                losses = []
                for seed in metadata["tuning_seeds"]:
                    _, _, _, info, _ = fit(samples, train, val, method, rate, seed,
                                            {**config, "steps": 300})
                    losses.append(info["validation_mse"])
                tuning.append({"method": method, "rate": rate, "validation_mse": losses})
                if all(loss is not None for loss in losses):
                    candidates.append((mean(losses), rate))
            if candidates:
                selected[method] = min(candidates)[1]
        for method in OPTIMIZERS:
            print(f"  {method}: {len(SEEDS)} seeds", flush=True)
            for seed in SEEDS:
                if method not in selected:
                    runs.append({"method": method, "seed": seed, "failed": True})
                    continue
                model, x, _, info, _ = fit(samples, train, val, method, selected[method], seed,
                    config, step_budgets=STEP_BUDGETS, time_budgets=TIME_BUDGETS)
                checkpoints = info.pop("checkpoints")
                for checkpoint in checkpoints.values():
                    model.load_state_dict(checkpoint.pop("state"))
                    prediction = predict(model, x, test)
                    loss = float(np.mean((samples.y[test] - prediction) ** 2))
                    checkpoint["test_mse"] = loss if np.isfinite(loss) else None
                runs.append({"method": method, "seed": seed, **info, "checkpoints": checkpoints})
        outcome = {"task": task, "chosen": selected, "tuning": tuning, "runs": runs}
        (args.out / f"{task}.json").write_text(json.dumps(outcome, indent=2, allow_nan=False) + "\n")
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    metadata["process_peak_rss_bytes"] = int(rss if platform.system() == "Darwin" else rss * 1024)
    (args.out / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
