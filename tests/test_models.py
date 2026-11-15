import numpy as np
import pytest
import torch

from loopalpha.data import toy_samples
from loopalpha.models import LoopModel
from loopalpha.optim import SeparateMoments, assign_depth_gradients, make_optimizers
from loopalpha.training import fit


def test_shared_gradient_equals_occurrence_sum_for_every_block_parameter():
    torch.manual_seed(3)
    x = torch.randn(7, 3, 2, dtype=torch.float64)
    model = LoopModel((3, 2), 8, 4).double()
    tied = model(x)
    assert tied.shape == (7,)
    tied.square().mean().backward()
    baseline = {key: p.grad.clone() for key, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    prediction, occurrences = model(x, probe=True)
    torch.testing.assert_close(tied, prediction)
    prediction.square().mean().backward()
    for key, p in model.blocks[0].named_parameters():
        assert p.grad is None
        total = sum(leaves[key].grad for leaves in occurrences)
        torch.testing.assert_close(total, baseline[f"blocks.0.{key}"], atol=1e-11, rtol=1e-9)
    for key, param in model.named_parameters():
        if not key.startswith("blocks."):
            torch.testing.assert_close(param.grad, baseline[key])


def test_tied_count_and_compute_matched_initialization():
    torch.manual_seed(7)
    tied = LoopModel((2, 2), 8, 4)
    shallow = LoopModel((2, 2), 8, 1)
    shallow.load_state_dict(tied.state_dict())
    torch.manual_seed(7)
    untied = LoopModel((2, 2), 8, 4, tied=False)
    x = torch.randn(5, 2, 2)
    assert sum(p.numel() for p in tied.parameters()) == sum(p.numel() for p in shallow.parameters())
    assert sum(p.numel() for p in untied.parameters()) > sum(p.numel() for p in tied.parameters())
    torch.testing.assert_close(tied(x), untied(x))
    assert not torch.allclose(tied(x), shallow(x))


@pytest.mark.parametrize("variant", ["mean", "separate"])
def test_depth_one_reduces_to_adamw(variant):
    torch.manual_seed(23)
    standard = LoopModel((2, 2), 8, 1).double()
    custom = LoopModel((2, 2), 8, 1).double()
    custom.load_state_dict(standard.state_dict())
    rate, decay = .003, .07
    standard_opt = torch.optim.AdamW(standard.parameters(), lr=rate, weight_decay=decay)
    if variant == "separate":
        custom_opt = torch.optim.AdamW([*custom.encode.parameters(), *custom.head.parameters()],
                                       lr=rate, weight_decay=decay)
        separate = SeparateMoments(custom, rate, decay)
    else:
        custom_opt = torch.optim.AdamW(custom.parameters(), lr=rate, weight_decay=decay)
    for _ in range(3):
        x = torch.randn(11, 2, 2, dtype=torch.float64)
        standard_opt.zero_grad()
        custom.zero_grad(set_to_none=True)
        standard(x).square().mean().backward()
        result, leaves = custom(x, probe=True)
        result.square().mean().backward()
        if variant == "separate":
            separate.assign(leaves)
            separate.step()
        else:
            assign_depth_gradients(custom, leaves)
        standard_opt.step()
        custom_opt.step()
        for actual, expected in zip(custom.parameters(), standard.parameters()):
            torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-8)


def test_muon_parameter_partition_is_disjoint_and_exhaustive():
    model = LoopModel((2, 2), 8, 4)
    muon, adam = make_optimizers(model, "muon", .001, .01)
    assert isinstance(muon, torch.optim.Muon)
    sets = [{id(p) for group in opt.param_groups for p in group["params"]} for opt in (muon, adam)]
    assert sets[0].isdisjoint(sets[1])
    assert sets[0] | sets[1] == {id(p) for p in model.parameters()}
    assert sets[0] == {id(p) for p in model.blocks.parameters() if p.ndim == 2}
    model(torch.randn(4, 2, 2)).square().mean().backward()
    for opt in (muon, adam):
        opt.step()
        assert opt.state_dict()["state"]
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_seeded_fit_is_reproducible_and_learns():
    toy = toy_samples(19, 50, 16, 3)
    config = {"width": 12, "steps": 50, "batch": 48, "depth": 2, "weight_decay": .01,
              "eval_every": 5, "time_budget": 1., "threshold": .2, "global_clip": 5.}
    idx = np.arange(600)
    holdout = np.arange(600, 800)
    test = lambda: fit(toy, idx, holdout, "adamw", .003, 7, config)[3]
    a, b = test(), test()
    assert a["validation_mse"] == b["validation_mse"]
    assert a["validation_mse"] < a["trace"][0]["val_mse"]


def test_depth_adamw_state_roundtrip():
    torch.manual_seed(3)
    original = LoopModel((3, 2), 8, 4)
    restored = LoopModel((3, 2), 8, 4)
    opt = make_optimizers(original, "adamw", .003, .01)[0]
    loaded = make_optimizers(restored, "adamw", .003, .01)[0]
    x = torch.randn(7, 3, 2)

    def advance(model, optimizer):
        optimizer.zero_grad()
        pred, leaves = model(x, probe=True)
        pred.square().mean().backward()
        assign_depth_gradients(model, leaves)
        optimizer.step()

    advance(original, opt)
    from copy import deepcopy
    restored.load_state_dict(original.state_dict())
    loaded.load_state_dict(deepcopy(opt.state_dict()))
    advance(original, opt)
    advance(restored, loaded)
    for actual, expected in zip(restored.parameters(), original.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_noise_control_keeps_model_input_dtype():
    toy = toy_samples(19, 20, 8, 3)
    config = {"width": 8, "steps": 2, "batch": 8, "depth": 2, "weight_decay": .01,
              "eval_every": 1, "time_budget": 1., "threshold": .2, "global_clip": 5., "data_seed": 19}
    _, x, _, result, _ = fit(toy, np.arange(80), np.arange(80, 100), "noise", .003, 7, config)
    assert x.dtype == torch.float32
    assert not result["failed"]
