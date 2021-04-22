from __future__ import annotations

import torch
import torch.nn as nn

from ..configs import Config


class ConvNet(nn.Module):
    """
    Basic convolutional neural network.
    
    Architecture adapted from https://www.kaggle.com/adityav5/cnn-on-fashion-mnist-approx-95-test-accuracy.
    """

    def __init__(self, cfg: Config):
        super().__init__()

        self.cfg = cfg

        C, H, W = self.cfg.experiment.input_size

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=C, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(p=0.3),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=128),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.3),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=256),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.3),
        )

        self.flatten = nn.Flatten(start_dim=1)

        self.dense = nn.Sequential(
            nn.Linear(in_features=256 * (H // 8) * (W // 8), out_features=2048),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=2048, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=self.cfg.experiment.n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        x = self.conv(x)
        x = self.flatten(x)
        x = self.dense(x)

        return x
