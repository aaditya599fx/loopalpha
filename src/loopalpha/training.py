"""Seeded CPU training and two evaluation budgets."""

from copy import deepcopy
from time import perf_counter

import numpy as np
import torch
from torch import nn

from .models import LoopModel, simple_model
from .optim import SeparateMoments, assign_depth_gradients, make_optimizers


def fit(samples, train, val, method, lr, seed, config):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    x = samples.x
    x_train = x[train]
    mean = x_train.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    scale = x_train.std(axis=(0, 1), dtype=np.float64).astype(np.float32)
    features = x
    labels = samples.y[train].copy()
    if method == "shuffle":
        dates = samples.date[train]
        for date in np.unique(dates):
            within = dates == date
            labels[within] = rng.permutation(labels[within])
    elif method == "noise":
        features = np.random.default_rng(config["data_seed"] + 1).normal(
            size=x.shape).astype(np.float32)
        mean, scale = 0., 1.
    batch_rng = np.random.default_rng(seed + 1)
    x = torch.from_numpy((features - mean) / np.maximum(scale, 1e-6))
    y = torch.from_numpy(labels)
    val_y = torch.from_numpy(samples.y[val])
    shape = x.shape[1:]
    if method in {"linear", "mlp"}:
        model = simple_model(shape, config["width"], method)
    else:
        depth = config["depth"]
        if method.startswith("k"):
            depth = int(method[1:])
        model = LoopModel(shape, config["width"], depth, tied=method != "untied")
    optimizer = method if method in {"sgd", "muon"} else "adamw"
    optimizers = make_optimizers(model, optimizer, lr, config["weight_decay"])
    depth_aware = method in {"depth", "depth_clip", "depth_late", "depth_moments"}
    separate = None
    if method == "depth_moments":
        separate = SeparateMoments(model, lr, config["weight_decay"])
        optimizers = [torch.optim.AdamW([*model.encode.parameters(), *model.head.parameters()],
                                        lr=lr, weight_decay=config["weight_decay"])]
    trace = []
    timed_state = None
    threshold_time = None
    threshold_step = None
    failed = False

    def validate():
        model.eval()
        with torch.no_grad():
            return nn.functional.mse_loss(model(x[val]), val_y).item()

    start = perf_counter()
    for step in range(1, config["steps"] + 1):
        model.train()
        indices = batch_rng.integers(0, len(train), size=config["batch"])
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        if depth_aware:
            prediction, occurrences = model(x[train[indices]], probe=True)
        else:
            prediction = model(x[train[indices]])
        loss = nn.functional.mse_loss(prediction, y[indices])
        if not torch.isfinite(loss):
            failed = True
            break
        loss.backward()
        if separate is not None:
            if not all(torch.isfinite(leaf.grad).all()
                       for leaves in occurrences for leaf in leaves.values()):
                failed = True
                break
            separate.assign(occurrences)
        elif depth_aware:
            assign_depth_gradients(model, occurrences,
                                   clip=config["clip"] if method != "depth" else None,
                                   weighting="late" if method == "depth_late" else "equal")
        try:
            nn.utils.clip_grad_norm_(model.parameters(), config["global_clip"], error_if_nonfinite=True)
        except RuntimeError:
            if all(param.grad is None or torch.isfinite(param.grad).all() for param in model.parameters()):
                raise
            failed = True
            break
        for opt in optimizers:
            opt.step()
        if separate is not None:
            separate.step()
        if step % config["eval_every"] == 0 or step == config["steps"]:
            val_loss = validate()
            elapsed = perf_counter() - start
            if not np.isfinite(val_loss):
                failed = True
                break
            trace.append({"step": step, "seconds": elapsed, "val_mse": val_loss})
            if elapsed <= config["time_budget"]:
                timed_state = (step, deepcopy(model.state_dict()))
            if threshold_time is None and val_loss <= config["threshold"]:
                threshold_time = elapsed
                threshold_step = step

    info = {"failed": failed, "seconds": perf_counter() - start,
            "parameters": sum(p.numel() for p in model.parameters()),
            "threshold_seconds": threshold_time, "threshold_step": threshold_step, "trace": trace,
            "validation_mse": trace[-1]["val_mse"] if trace and not failed else None,
            "probe_parameter_bytes": model.depth * sum(p.numel() * p.element_size()
                                                       for p in model.blocks[0].parameters())
            if depth_aware else 0}
    transform = {"mean": torch.as_tensor(mean), "scale": torch.as_tensor(np.maximum(scale, 1e-6))}
    return model, x, timed_state, info, transform
