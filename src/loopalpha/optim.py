"""Baseline grouping and the experimental direction supplied to AdamW."""

import torch


def make_optimizers(model, method, lr, decay):
    if method == "sgd":
        return [torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=decay)]
    if method == "muon":
        matrices = [param for name, param in model.named_parameters()
                    if param.ndim == 2 and name.startswith("blocks.")]
        eligible = {id(param) for param in matrices}
        remainder = [param for param in model.parameters() if id(param) not in eligible]
        return [torch.optim.Muon(matrices, lr=lr, weight_decay=decay),
                torch.optim.AdamW(remainder, lr=lr, weight_decay=decay)]
    return [torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=decay)]


def assign_depth_gradients(model, occurrences, *, clip=None, weighting="equal"):
    """Replace shared-block gradients after one backward pass through probe leaves."""
    device = next(model.parameters()).device
    weights = torch.ones(len(occurrences), device=device)
    if weighting == "late":
        weights = torch.arange(1, len(occurrences) + 1, device=device)
    elif weighting != "equal":
        raise ValueError(f"Unknown weighting {weighting}")
    weights = weights / weights.sum()
    scales = [1.0] * len(occurrences)
    if clip is not None:
        for i, leaves in enumerate(occurrences):
            norm = torch.linalg.vector_norm(torch.stack(
                [torch.linalg.vector_norm(leaf.grad) for leaf in leaves.values()]))
            scales[i] = (clip / norm.clamp_min(1e-12)).clamp_max(1)
    for name, param in model.blocks[0].named_parameters():
        param.grad = torch.stack([scale * weight * leaves[name].grad
                                  for scale, weight, leaves in zip(scales, weights, occurrences)]).sum(0)


class SeparateMoments:
    """Average bias-corrected Adam directions, one moment pair per occurrence."""

    def __init__(self, model, rate, decay):
        self.parameters = dict(model.blocks[0].named_parameters())
        self.rate = rate
        self.decay = decay
        self.iteration = 0
        self.moments = [{name: (torch.zeros_like(p), torch.zeros_like(p))
                         for name, p in self.parameters.items()} for _ in range(model.depth)]

    @torch.no_grad()
    def assign(self, occurrences):
        self.iteration += 1
        bias1 = 1 - .9 ** self.iteration
        bias2 = 1 - .999 ** self.iteration
        for name, param in self.parameters.items():
            directions = []
            for leaves, state in zip(occurrences, self.moments):
                grad = leaves[name].grad
                first, second = state[name]
                first.lerp_(grad, .1)
                second.mul_(.999).addcmul_(grad, grad, value=.001)
                directions.append((first / bias1) / ((second / bias2).sqrt() + 1e-8))
            param.grad = torch.stack(directions).mean(0)

    @torch.no_grad()
    def step(self):
        for param in self.parameters.values():
            param.mul_(1 - self.rate * self.decay).add_(param.grad, alpha=-self.rate)
