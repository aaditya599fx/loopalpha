"""Validation-only model selection, walk-forward scoring and provenance."""

import argparse
import json
import platform
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from statistics import mean

import numpy as np
import pandas as pd
import torch
import yaml

from .data import market_samples, read_panel, read_sessions, synthetic_panel, toy_samples, validate_panel
from .evaluation import block_interval, correlations, cost_metrics, portfolio
from .training import fit


OPTIMIZERS = ("sgd", "adamw", "muon", "depth", "depth_clip", "depth_late", "depth_moments")
BASELINES = ("linear", "mlp", "untied")
CONTROLS = ("shuffle", "noise")
DEPTHS = ("k1", "k2", "k8")


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def predict(model, x, indices):
    model.eval()
    with torch.inference_mode():
        return torch.cat([model(x[part]) for part in np.array_split(indices, max(1, len(indices) // 4096))]
                         ).numpy()


def finite_indices(samples, start, stop):
    indices = samples.split(start, stop)
    return indices[np.isfinite(samples.y[indices])]


def evaluate(samples, indices, prediction, directory, start, stop, config):
    if not np.isfinite(prediction).all():
        return {"status": "nonfinite_prediction"}
    valid = np.isfinite(samples.y[indices])
    result = {"test_mse": float(np.mean((prediction[valid] - samples.y[indices][valid]) ** 2)),
              "test_examples": int(valid.sum())}
    if samples.dates is None:
        return result
    dates = samples.date[indices]
    frame = pd.DataFrame({"session": dates, "date": samples.dates[dates],
                          "symbol": samples.symbol[indices], "prediction": prediction,
                          "realized": samples.realized[indices]})
    frame.to_csv(directory / "predictions.csv", index=False)
    daily = correlations(frame)
    daily.to_csv(directory / "daily_ic.csv", index=False)
    result["ic_days"] = len(daily)
    result["rank_ic_95"] = block_interval(daily.rank_ic, block=max(5, samples.horizon))
    if len(daily):
        result.update({"ic": float(daily.ic.mean()), "rank_ic": float(daily.rank_ic.mean()),
                       "rank_ic_positive": float(daily.rank_ic.gt(0).mean()),
                       "rank_ic_std": float(daily.rank_ic.std(ddof=0)),
                       "rank_ic_ir": float(daily.rank_ic.mean() / daily.rank_ic.std(ddof=0))
                       if daily.rank_ic.std(ddof=0) > 0 else None})
    ledger = portfolio(frame, samples.opens, samples.horizon, start, stop)
    ledger.to_csv(directory / "trade_ledger.csv", index=False)
    result["cost_stress"] = {str(bps): cost_metrics(ledger, samples.horizon, bps, start, stop)
                             for bps in config["cost_bps"]}
    return result


def fold(samples, boundaries, config, output):
    train_stop, val_stop, test_stop = boundaries
    train = finite_indices(samples, 0, train_stop)
    val = finite_indices(samples, train_stop, val_stop)
    labeled_test = finite_indices(samples, val_stop, test_stop)
    test = samples.split(val_stop, test_stop)
    if not min(len(train), len(val), len(labeled_test)):
        raise ValueError(f"Empty labeled partition for session boundaries {boundaries}")
    chosen = {}
    trials = []
    methods = OPTIMIZERS + BASELINES + (CONTROLS if samples.dates is not None else DEPTHS)
    for method in methods:
        if method in CONTROLS or method in DEPTHS:
            rates = [chosen["adamw"]]
        else:
            rates = config["grids"].get(method, config["grids"]["adamw"])
        options = []
        for rate in rates:
            losses = []
            for seed in config["seeds"]:
                _, _, _, info, _ = fit(samples, train, val, method, rate, seed, config)
                trials.append({"method": method, "rate": rate, "seed": seed, **info})
                losses.append(info["validation_mse"])
            if all(loss is not None for loss in losses):
                options.append((mean(losses), rate))
        if options:
            chosen[method] = min(options)[1]
        if method == "adamw" and not options:
            write_json(output / "tuning.json", trials)
            raise RuntimeError("AdamW diverged at every validation candidate")
    write_json(output / "tuning.json", trials)
    results = []
    for method in methods:
        for seed in config["seeds"]:
            directory = output / method / str(seed)
            directory.mkdir(parents=True)
            if method not in chosen:
                results.append({"method": method, "seed": seed, "failed": True})
                continue
            row = {"method": method, "seed": seed, "rate": chosen[method]}
            model, x, timed, info, transform = fit(samples, train, val, method, chosen[method], seed, config)
            row.update(info)
            if not info["failed"]:
                row.update(evaluate(samples, test, predict(model, x, test), directory,
                                    val_stop, test_stop, config))
                torch.save({"state_dict": model.state_dict(), "transform": transform,
                            "input_shape": tuple(x.shape[1:]), "width": config["width"],
                            "depth": getattr(model, "depth", None), "method": method},
                           directory / "model.pt")
                if timed is not None:
                    time_step, state = timed
                    model.load_state_dict(state)
                    pred = predict(model, x, labeled_test)
                    row["time_budget_step"] = time_step
                    row["time_budget_test_mse"] = float(np.mean((pred - samples.y[labeled_test]) ** 2))
            results.append(row)
    if samples.dates is not None:
        momentum = output / "momentum"
        momentum.mkdir()
        row = evaluate(samples, test, samples.signal[test], momentum, val_stop, test_stop, config)
        row.pop("test_mse", None)
        results.append({"method": "momentum", **row})
    write_json(output / "results.json", {"boundaries": boundaries, "chosen": chosen,
                                         "counts": {"train": len(train), "validation": len(val),
                                                    "labeled_test": len(labeled_test), "test": len(test)},
                                         "runs": results})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=["toy", "synthetic-market", "market"])
    parser.add_argument("--config", default="configs/research.yaml", type=Path)
    parser.add_argument("--out", required=True, type=Path, help="New output directory")
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--calendar", type=Path, help="Optional market CSV with ordered date column")
    parser.add_argument("--source", help="Market-data provenance and license")
    parser.add_argument("--prices", choices=["adjusted", "unadjusted"])
    args = parser.parse_args(argv)
    if args.calendar and args.task != "market":
        parser.error("--calendar is supported for the market task")
    config_bytes = args.config.read_bytes()
    config = yaml.safe_load(config_bytes)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    seed, sessions, assets = (config[key] for key in ("data_seed", "sessions", "assets"))
    audit = {"source": args.task, "seed": seed}
    if args.task == "toy":
        samples = toy_samples(seed, sessions, assets, 5)
    else:
        if args.task == "market":
            if not (args.csv and args.source and args.prices):
                parser.error("market requires --csv, --source and --prices")
            panel, audit = read_panel(args.csv, source=args.source, adjusted=args.prices == "adjusted")
        else:
            panel = validate_panel(synthetic_panel(seed, sessions, assets))
        sessions, calendar = None, {"calendar": "union of supplied sessions"}
        if args.calendar:
            sessions, calendar = read_sessions(args.calendar, panel)
        samples = market_samples(panel, config["lookback"], config["horizon"], sessions=sessions)
        audit.update({"raw_rows": len(panel), "eligible_feature_rows": len(samples.x),
                      "valid_labeled_rows": int(np.isfinite(samples.y).sum()),
                      "sessions": samples.sessions, **calendar,
                      "point_in_time_eligibility_field": "eligible" in panel})
    args.out.mkdir(parents=True)
    source_files = sorted(Path(__file__).parent.glob("*.py"))
    write_json(args.out / "manifest.json", {"task": args.task, "data": audit, "config": config,
               "config_sha256": sha256(config_bytes).hexdigest(),
               "code_sha256": {p.name: sha256(p.read_bytes()).hexdigest() for p in source_files},
               "utc": datetime.now(timezone.utc).isoformat(),
               "platform": platform.platform(), "python": platform.python_version(),
               "torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__,
               "device": "cpu", "threads": 1})
    n = samples.sessions
    splits = [(int(.6 * n), int(.8 * n), n)] if args.task == "toy" else [
        (int(.45 * n), int(.60 * n), int(.75 * n)),
        (int(.60 * n), int(.75 * n), n)]
    for i, boundaries in enumerate(splits, 1):
        directory = args.out / f"fold-{i}"
        directory.mkdir()
        fold(samples, boundaries, config, directory)


if __name__ == "__main__":
    main()
