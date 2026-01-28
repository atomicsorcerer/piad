import torch
import torch.nn as nn


class ParticleFlowNetwork(nn.Module):
    def __init__(
        self,
        input_size: int,
        total_feature_size: int,
        latent_space_dim: int,
        classifier_hidden_layer_dimensions: list[int],
        mapping_hidden_layer_dimensions: list[int],
    ) -> None:
        """
        Particle Flow Network model, based off of the Deep Sets framework. Takes per-particle information and attempts
        to classify the event as signal or background.

        Args:
                input_size: The number of observables per particle.
                total_feature_size: Nuber of total features (including all particles).
                latent_space_dim: The size of the latent space vector.
                classifier_hidden_layer_dimensions:Array with the size of each linear layer for the classification.
                mapping_hidden_layer_dimensions: Array with the size of each linear layer for the mapping.

        Raises:
                ValueError: Length of classifier_hidden_layer_dimensions cannot be zero.
        """
        super().__init__()

        self.latent_space_dim = latent_space_dim

        if len(classifier_hidden_layer_dimensions) == 0:
            raise ValueError(
                "Length of classifier_hidden_layer_dimensions cannot be zero."
            )

        stack = nn.Sequential(
            nn.Linear(latent_space_dim, classifier_hidden_layer_dimensions[0]),
            nn.BatchNorm1d(classifier_hidden_layer_dimensions[0]),
            nn.ReLU(),
        )

        for i in range(len(classifier_hidden_layer_dimensions)):
            stack.append(
                nn.Linear(
                    classifier_hidden_layer_dimensions[i],
                    (
                        classifier_hidden_layer_dimensions[i]
                        if i == len(classifier_hidden_layer_dimensions) - 1
                        else classifier_hidden_layer_dimensions[i + 1]
                    ),
                )
            )
            stack.append(
                nn.BatchNorm1d(
                    classifier_hidden_layer_dimensions[i]
                    if i == len(classifier_hidden_layer_dimensions) - 1
                    else classifier_hidden_layer_dimensions[i + 1]
                )
            )
            stack.append(nn.ReLU())

        stack.append(nn.Linear(classifier_hidden_layer_dimensions[-1], 1))

        self.stack = stack
        self.p_map = ParticleMapping(
            input_size,
            total_feature_size,
            latent_space_dim,
            mapping_hidden_layer_dimensions,
        )

    def forward(self, x) -> torch.Tensor:
        """
        Forward implementation for ParticleFlowNetwork.

        Args:
                x: Input tensor(s).

        Returns:
                torch.Tensor: Output tensor with two values each representing the probabilities of signal and background.
        """
        # rapidity = 0.5 * (torch.log((x[..., 3] + x[..., 2]) / (x[..., 3] - x[..., 2])))
        # print(x)
        # Need to figure out how to deal with cases where p_z is more negative than energy is positive
        # print(((x[..., 3] < torch.abs(x[..., 2]))))
        # x = torch.sign(x) * torch.log(torch.abs(x) + 1)

        # pT = torch.sqrt(torch.add(torch.pow(x[..., 0][..., 0], 2), torch.pow(x[..., 1][..., 0], 2)))
        # rapidity = 0.5 * (torch.log((x[..., 3] + x[..., 2]) / (x[..., 3] - x[..., 2])))
        # x = torch.sign(x) * torch.log(torch.abs(x) + 1)
        x = self.p_map(x)
        x = self.stack(x)

        return x


class ParticleMapping(nn.Module):
    def __init__(
        self,
        input_size: int,
        total_features: int,
        output_dimension: int,
        hidden_layer_dimensions=None,
    ) -> None:
        """
        Maps each set of observables of a particle to a specific dimensional output and sums them together.

        Args:
                input_size: The number of data points each particle has.
                output_dimension: The fixed number of output nodes.
                hidden_layer_dimensions: A list of numbers which set the sizes of hidden layers.

        Raises:
                TypeError: If hidden_layer_dimensions is not a list.
                ValueError: If hidden_layer_dimensions is an empty list.
                ValueError: Input tensor must be able to evenly split for the given input size.
        """
        super().__init__()

        self.input_size = input_size
        self.total_features = total_features
        self.output_dimension = output_dimension

        if hidden_layer_dimensions is None:
            hidden_layer_dimensions = [100]
        elif not isinstance(hidden_layer_dimensions, list):
            raise TypeError(
                f"Hidden layer dimensions must be a valid list. {hidden_layer_dimensions} is not valid."
            )
        elif len(hidden_layer_dimensions) == 0:
            raise ValueError("Hidden layer dimensions cannot be empty.")

        if total_features % input_size != 0:
            raise ValueError(
                f"Each particle must have the same number of observables, which must be equal to the input size. "
                f"Total_features % input_size must be zero."
            )

        stack = nn.Sequential(
            nn.Linear(input_size, hidden_layer_dimensions[0]),
            nn.BatchNorm1d(total_features // input_size),
            nn.ReLU(),
        )

        for i in range(len(hidden_layer_dimensions)):
            stack.append(
                nn.Linear(
                    hidden_layer_dimensions[i],
                    (
                        hidden_layer_dimensions[i]
                        if i == len(hidden_layer_dimensions) - 1
                        else hidden_layer_dimensions[i + 1]
                    ),
                )
            )
            stack.append(nn.BatchNorm1d(total_features // input_size))
            stack.append(nn.ReLU())

        stack.append(nn.Linear(hidden_layer_dimensions[-1], output_dimension))

        self.stack = stack
        self.avg_pool_2d = nn.AvgPool2d((total_features // input_size, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward implementation for ParticleMapping.

        Args:
                x: Input tensor(s).

        Returns:
                torch.Tensor: Output tensor with predefined dimensions.
        """
        x = self.stack(x)
        x = self.sum_pool_2d(x)
        x = torch.squeeze(x, 1)

        return x

    def sum_pool_2d(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs sum pooling operation.

        Args:
                x: Input tensor(s).

        Returns:
                torch.Tensor: Output tensor with predefined output dimensions.
        """
        x = self.avg_pool_2d(x)
        x = torch.mul(x, self.total_features // self.input_size)

        return x
