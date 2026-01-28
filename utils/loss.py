import torch
from matplotlib import pyplot as plt
from nflows.flows import Flow

from .physics import calculate_squared_dijet_mass


def calculate_outlier_gradient_penalty_with_preprocess_mod_z_scores(
    log_prob: torch.Tensor,
    inputs: torch.Tensor,
    median: float,
    mad: float,
    loss_cut_off: float = 2.0,
    alpha: float = 1.0,
    z_score_factor: float = 1.0,
):
    grad_log_prob_first_order = torch.autograd.grad(
        outputs=log_prob.sum(), inputs=inputs, create_graph=True
    )[0]
    gradients = torch.norm(grad_log_prob_first_order, dim=1)
    modified_z_scores = 0.6745 * (gradients - median) / mad
    penalty = torch.nn.functional.relu(
        modified_z_scores * z_score_factor - loss_cut_off
    )
    penalty = torch.mean(penalty)
    penalty = penalty * alpha

    return penalty


def calculate_outlier_gradient_penalty_with_preprocess(
    log_prob: torch.Tensor,
    inputs: torch.Tensor,
    std_dev: float,
    mean: float,
    loss_cut_off: float = 2.0,
    alpha: float = 1.0,
):
    grad_log_prob_first_order = torch.autograd.grad(
        outputs=log_prob.sum(), inputs=inputs, create_graph=True
    )[0]
    grad_log_prob_second_order = torch.autograd.grad(
        outputs=grad_log_prob_first_order.sum(), inputs=inputs, create_graph=True
    )[0]
    gradients = torch.norm(grad_log_prob_second_order, dim=1)
    grad_z_scores = (gradients - mean) / std_dev
    penalty = torch.nn.functional.relu(grad_z_scores - loss_cut_off)
    penalty = torch.mean(penalty)
    penalty = penalty * alpha

    return penalty


def calculate_outlier_gradient_penalty(
    log_prob: torch.Tensor,
    inputs: torch.Tensor,
    cut_off: float = 4.0,
    alpha: float = 1.0,
) -> torch.Tensor:
    grad_log_prob = torch.autograd.grad(
        outputs=log_prob.sum(), inputs=inputs, create_graph=True
    )[0]
    gradients = torch.norm(grad_log_prob, dim=1)
    grad_mean = torch.mean(gradients)
    grad_std_dev = torch.std(gradients)
    grad_z_scores = (gradients - grad_mean) / grad_std_dev
    penalty = torch.abs(grad_z_scores)
    penalty = torch.nn.functional.relu(penalty - cut_off)
    penalty = torch.mean(penalty)
    penalty = penalty * alpha

    return penalty


def calculate_first_order_non_smoothness_penalty(
    log_prob: torch.Tensor, inputs: torch.Tensor, alpha: float
) -> torch.Tensor:
    grad_log_prob = torch.autograd.grad(
        outputs=log_prob.sum(), inputs=inputs, create_graph=True
    )[0]
    smoothness_penalty = torch.norm(grad_log_prob, dim=1)
    smoothness_penalty = torch.mean(smoothness_penalty)
    smoothness_penalty = smoothness_penalty * alpha

    return smoothness_penalty


def calculate_second_order_non_smoothness_penalty(
    log_prob: torch.Tensor, inputs: torch.Tensor, alpha: float
) -> torch.Tensor:
    grad_log_prob = torch.autograd.grad(
        outputs=log_prob.sum(), inputs=inputs, create_graph=True, retain_graph=True
    )[0]
    second_order_grad_log_prob = torch.autograd.grad(
        outputs=grad_log_prob.sum(), inputs=inputs, create_graph=True, retain_graph=True
    )[0]
    smoothness_penalty = torch.norm(second_order_grad_log_prob, dim=1)
    smoothness_penalty = torch.mean(smoothness_penalty)
    smoothness_penalty = smoothness_penalty * alpha

    return smoothness_penalty


def calculate_knotted_non_smoothness_penalty(densities: torch.Tensor, n_knots: int):
    slopes = (densities[1:] - densities[:-1]) * (n_knots - 1)
    # slopes = slopes.abs()
    slope_differences = slopes[1:] - slopes[:-1]
    slope_differences = slope_differences.abs()
    smoothness_penalty = slope_differences.mean()
    smoothness_penalty = smoothness_penalty

    return smoothness_penalty


def calculate_knotted_non_smoothness_penalty_1d(
    flow: Flow, start: float, end: float, n_knots: int, alpha: float
) -> torch.Tensor:
    knots = torch.linspace(start, end, n_knots).reshape((-1, 1))
    densities = flow.log_prob(knots).exp()
    smoothness_penalty = calculate_knotted_non_smoothness_penalty(densities, n_knots)
    smoothness_penalty = smoothness_penalty * alpha

    return smoothness_penalty


def calculate_knotted_marginal_non_smoothness_penalty_multi_dim(
    flow: Flow, start: float, end: float, n_knots: int, n_dims: int, alpha: float
) -> torch.Tensor:
    smoothness_penalty = torch.tensor(0.0)
    knots_1d = torch.linspace(start, end, n_knots)
    for i in range(n_dims):
        knots = torch.zeros(n_dims, n_knots)
        knots[i] += knots_1d
        knots = torch.transpose(knots, 0, 1)
        densities = flow.log_prob(knots).exp()
        ind_smoothness_penalty = calculate_knotted_non_smoothness_penalty(
            densities, n_knots
        )
        smoothness_penalty += ind_smoothness_penalty

    return smoothness_penalty * alpha


def calculate_impossible_mass_penalty(
    flow: Flow, n_samples: int, alpha: float
) -> torch.Tensor:
    base_samples = flow._distribution.sample(n_samples)
    generated_samples, _ = flow._transform.inverse(base_samples)
    generated_masses = calculate_squared_dijet_mass(generated_samples)
    negative_mass_penalty = (
        torch.nn.functional.softplus(-generated_masses).mean() * alpha
    )

    return negative_mass_penalty
