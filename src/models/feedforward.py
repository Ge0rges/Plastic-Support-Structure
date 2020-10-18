import torch.nn as nn
import torch
import numpy as np


class FeedForward(nn.Module):
    def __init__(self, sizes: dict, old_weights=None, old_biases=None):
        super(FeedForward, self).__init__()

        self.phase = None  # irrelevant for this model

        self.sizes = sizes
        self.input_size = sizes["classifier"][0]

        # Classifier
        ff_layers = self.set_module('classifier', old_weights=old_weights, old_biases=old_biases)
        ff_layers.append(nn.Sigmoid())  # Must be non-linear
        self.classifier = nn.Sequential(*ff_layers)

        # Make float
        self.float()

    def forward(self, x):
        x = self.classifier(x)
        return x

    def set_module(self, label, old_weights=None, old_biases=None):
        sizes = self.sizes[label]

        if old_weights:
            old_weights = old_weights[label]

        if old_biases:
            old_biases = old_biases[label]

        layers = [
            self.get_layer(sizes[0], sizes[1], old_weights, old_biases, 0)
        ]

        for i in range(1, len(sizes) - 1):
            layers.append(nn.LeakyReLU())
            layers.append(self.get_layer(sizes[i], sizes[i+1], init_weights=old_weights, init_biases=old_biases, index=i))

        return layers

    def get_layer(self, input, output, init_weights=None, init_biases=None, index=0):
        layer = nn.Linear(input, output)
        torch.nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='leaky_relu')

        if init_weights is not None:
            weights = init_weights[index]

            # Type checking
            if isinstance(weights, list):
                weights = np.asarray(weights, dtype=float)

            if isinstance(weights, np.ndarray):
                weights = torch.from_numpy(weights)

            if isinstance(weights, torch.Tensor):
                weights = nn.Parameter(weights)

            if isinstance(weights, torch.nn.Parameter):
                layer.weight = weights

            # Padding
            weights = layer.weight.detach()

            if input > weights.shape[1]:
                kaiming_weights = torch.rand(weights.shape[0], input - weights.shape[1]).to(weights.device)
                torch.nn.init.kaiming_uniform_(kaiming_weights, mode='fan_in', nonlinearity='leaky_relu')

                weights = torch.cat([weights.float(), kaiming_weights.float()], dim=1)

            if output > weights.shape[0]:
                kaiming_weights = torch.rand(output - weights.shape[0], input).to(weights.device)
                torch.nn.init.kaiming_uniform_(kaiming_weights, mode='fan_in', nonlinearity='leaky_relu')

                weights = torch.cat([weights.float(), kaiming_weights.float()], dim=0)

            # Set
            layer.weight = nn.Parameter(weights)

        if init_biases is not None:
            biases = init_biases[index]

            # Type checking
            if isinstance(biases, list):
                biases = np.asarray(biases, dtype=float)

            if isinstance(biases, np.ndarray):
                biases = torch.from_numpy(biases)

            if isinstance(biases, torch.Tensor):
                biases = nn.Parameter(biases)

            if isinstance(biases, torch.nn.Parameter):
                layer.bias = biases

            # Padding
            biases = layer.bias.detach()

            if output != biases.shape[0]:
                rand_biases = torch.rand(output - biases.shape[0]).to(biases.device)

                biases = torch.cat([biases.float(), rand_biases.float()], dim=0)

            # Set
            layer.bias = nn.Parameter(biases)

            # Update the old_biases to include padding
            init_biases[index] = layer.bias.detach()

        return layer.float()
