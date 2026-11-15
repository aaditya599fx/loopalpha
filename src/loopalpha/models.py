"""Sequence encoders and exact per-occurrence derivatives."""

import torch
from torch import nn
from torch.func import functional_call


class ResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))
        self.norm = nn.LayerNorm(width)

    def forward(self, state):
        return self.norm(state + self.layers(state))


class LoopModel(nn.Module):
    def __init__(self, shape, width, depth, tied=True):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be positive")
        self.depth = depth
        self.tied = tied
        self.encode = nn.Sequential(nn.Flatten(), nn.Linear(shape[0] * shape[1], width), nn.GELU())
        self.head = nn.Linear(width, 1)
        self.blocks = nn.ModuleList(ResidualBlock(width) for _ in range(1 if tied else depth))
        if not tied:
            for block in self.blocks[1:]:
                block.load_state_dict(self.blocks[0].state_dict())

    def forward(self, sequence, *, probe=False):
        if probe and not self.tied:
            raise ValueError("depth probing requires tied weights")
        state = self.encode(sequence)
        occurrences = []
        for depth in range(self.depth):
            block = self.blocks[0 if self.tied else depth]
            if probe:
                leaves = {key: value.detach().clone().requires_grad_()
                          for key, value in block.named_parameters()}
                state = functional_call(block, leaves, (state,), strict=True)
                occurrences.append(leaves)
            else:
                state = block(state)
        prediction = self.head(state).squeeze(-1)
        return (prediction, occurrences) if probe else prediction


def simple_model(shape, width, method):
    inputs = shape[0] * shape[1]
    if method == "linear":
        return nn.Sequential(nn.Flatten(), nn.Linear(inputs, 1), nn.Flatten(start_dim=0))
    return nn.Sequential(nn.Flatten(), nn.Linear(inputs, width), nn.GELU(),
                         nn.Linear(width, 1), nn.Flatten(start_dim=0))
