import torch
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self, input_channels, output_channels, hidden_units):
        super().__init__()
        self.conv_layer_1 = self._create_conv_layer(kernel_size=3, input_channels=input_channels, output_channels=hidden_units,
                                                    stride=2, hidden_units=hidden_units, max_pool_kernel=3, padding=1)
        self.conv_layer_2 = self._create_conv_layer(kernel_size=3, input_channels=hidden_units, output_channels=8,
                                                    stride=1, hidden_units=hidden_units, max_pool_kernel=2,padding=0)
        self.dark_net = self._create_dark_net(output_channels)

    def _create_conv_layer(self, kernel_size, input_channels, output_channels, stride, hidden_units, max_pool_kernel,
                           padding=0):
        return nn.Sequential(
            nn.Conv2d(kernel_size=kernel_size, in_channels=input_channels, out_channels=hidden_units, stride=stride,
                      padding=padding),
            nn.ELU(),
            nn.Conv2d(kernel_size=kernel_size, in_channels=hidden_units, out_channels=output_channels, stride=stride,
                      padding=padding),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=max_pool_kernel)
        )

    def _create_dark_net(self, output_channels):
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, out_features=output_channels)
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.dark_net(self.conv_layer_2(self.conv_layer_1(x)))
