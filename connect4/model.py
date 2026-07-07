"""The policy + value network: a small residual conv net (AlphaZero-style).

Input : (batch, 2, N_ROWS, N_COLS) board planes from the mover's perspective.
Output: policy logits over the N_COLS columns, and a scalar value in [-1, 1]
        estimating the game result for the side to move.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import BLOCKS, CHANNELS, N_COLS, N_ROWS


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(x + y)


class Connect4Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(2, CHANNELS, 3, padding=1, bias=False),
            nn.BatchNorm2d(CHANNELS),
            nn.ReLU(),
        )
        self.tower = nn.Sequential(*[ResidualBlock(CHANNELS) for _ in range(BLOCKS)])

        # Policy head.
        self.policy_conv = nn.Conv2d(CHANNELS, 2, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * N_ROWS * N_COLS, N_COLS)

        # Value head.
        self.value_conv = nn.Conv2d(CHANNELS, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(N_ROWS * N_COLS, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.tower(self.stem(x))

        p = F.relu(self.policy_bn(self.policy_conv(x)))
        policy_logits = self.policy_fc(p.flatten(1))

        v = F.relu(self.value_bn(self.value_conv(x)))
        v = F.relu(self.value_fc1(v.flatten(1)))
        value = torch.tanh(self.value_fc2(v)).squeeze(1)

        return policy_logits, value
