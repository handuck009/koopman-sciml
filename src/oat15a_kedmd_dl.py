"""Kernel dictionary learning for POD-reduced OAT15A buffet dynamics.

The implementation follows Algorithm 2 in Bolager et al. (2026), Dictionary
Learning for Kernel EDMD. It uses the simplified kernel EDMD (skEDMD)
dictionary g(., centers), analytically updates the Koopman and state decoder
matrices, and updates a parameterized kernel mixture with prediction loss.
"""

from __future__ import annotations

import copy
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from oat15a_edmd_dl import pressure_from_states, relative_error
from oat15a_pod import PodConfig, load_oat15a_data, make_triangulation, resolve_data_dir


@dataclass
class KernelDLConfig:
    """Configuration for the OAT15A skEDMD-DL experiment."""

    data_root: str = "F:/MY JOURNEY/data-driven_model"
    data_dir: str | None = None
    pod_dmd_results: str = (
        "E:/Agent/koopman_sciml/results/oat15a_pod_dmd_stride2/"
        "oat15a_stride2_pod_dmd_results.npz"
    )
    stride: int = 2
    state_rank: int = 5
    training_fraction: float = 0.60
    validation_fraction: float = 0.10
    center_count: int = 80
    batch_count: int = 5
    learning_rate: float = 5e-3
    max_epochs: int = 200
    validation_interval: int = 5
    koopman_regularization_initial: float = 1e-3
    koopman_regularization_final: float = 1e-6
    evaluation_koopman_regularization: float = 1e-3
    regularization_switch_epoch: int = 30
    modes_regularization: float = 1e-6
    evaluation_modes_regularization: float = 1e-4
    outer_l1_regularization: float = 1e-8
    inner_l2_regularization: float = 1e-8
    gradient_clip: float = 10.0
    random_seed: int = 19
    rbf_sigmas: tuple[float, float, float] = (0.5, 2.0, 10.0)
    cosine_frequency: float = 1.0
    linear_scale: float = 1.0
    nngp_weight_scale: float = 1.0
    nngp_bias_scale: float = 0.1
    reduced_frequency_factor: float = float(np.pi / 2.0)
    dt: float = 0.07295943052
    zoom_xlim: tuple[float, float] = (-0.2, 1.2)
    zoom_ylim: tuple[float, float] = (-0.4, 0.4)


def set_random_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def inverse_softplus(value: float) -> float:
    value = max(float(value), 1e-8)
    return float(np.log(np.expm1(value)))


def pairwise_squared_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Return row-wise squared Euclidean distances."""

    x_norm = torch.sum(x * x, dim=1, keepdim=True)
    y_norm = torch.sum(y * y, dim=1, keepdim=True).T
    return torch.clamp(x_norm + y_norm - 2.0 * x @ y.T, min=0.0)


class LearnedKernelMixture(nn.Module):
    """Weighted RBF/cosine/linear/NNGP kernel family from the paper."""

    component_names = (
        "RBF sigma 0.5",
        "RBF sigma 2",
        "RBF sigma 10",
        "cosine",
        "linear",
        "ReLU NNGP",
    )

    def __init__(self, config: KernelDLConfig) -> None:
        super().__init__()
        self.raw_outer_weights = nn.Parameter(torch.full((6,), 0.25, dtype=torch.float64))
        self.raw_rbf_sigmas = nn.Parameter(
            torch.tensor([inverse_softplus(value) for value in config.rbf_sigmas], dtype=torch.float64)
        )
        self.raw_cosine_frequency = nn.Parameter(
            torch.tensor(inverse_softplus(config.cosine_frequency), dtype=torch.float64)
        )
        self.raw_linear_scale = nn.Parameter(
            torch.tensor(inverse_softplus(config.linear_scale), dtype=torch.float64)
        )
        self.raw_nngp_weight = nn.Parameter(
            torch.tensor(inverse_softplus(config.nngp_weight_scale), dtype=torch.float64)
        )
        self.raw_nngp_bias = nn.Parameter(
            torch.tensor(inverse_softplus(config.nngp_bias_scale), dtype=torch.float64)
        )

    @staticmethod
    def positive(raw: torch.Tensor) -> torch.Tensor:
        return F.softplus(raw) + 1e-6

    def normalized_outer_weights(self) -> torch.Tensor:
        denominator = torch.sum(torch.abs(self.raw_outer_weights)) + 1e-12
        return self.raw_outer_weights / denominator

    def mixture_coefficients(self) -> torch.Tensor:
        return self.normalized_outer_weights() ** 2

    def internal_parameters(self) -> dict[str, torch.Tensor]:
        sigmas = self.positive(self.raw_rbf_sigmas)
        return {
            "rbf_sigma_1": sigmas[0],
            "rbf_sigma_2": sigmas[1],
            "rbf_sigma_3": sigmas[2],
            "cosine_frequency": self.positive(self.raw_cosine_frequency),
            "linear_scale": self.positive(self.raw_linear_scale),
            "nngp_weight_scale": self.positive(self.raw_nngp_weight),
            "nngp_bias_scale": self.positive(self.raw_nngp_bias),
        }

    def components(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        params = self.internal_parameters()
        distance_squared = pairwise_squared_distance(x, y)
        rbf = [
            torch.exp(-distance_squared / (2.0 * params[f"rbf_sigma_{index + 1}"] ** 2))
            for index in range(3)
        ]
        cosine = torch.cos(params["cosine_frequency"] * distance_squared)
        dimension = float(x.shape[1])
        linear = params["linear_scale"] ** 2 * (x @ y.T) / dimension

        weight_scale = params["nngp_weight_scale"]
        bias_scale = params["nngp_bias_scale"]
        covariance = weight_scale**2 * (x @ y.T) / dimension + bias_scale**2
        variance_x = weight_scale**2 * torch.sum(x * x, dim=1) / dimension + bias_scale**2
        variance_y = weight_scale**2 * torch.sum(y * y, dim=1) / dimension + bias_scale**2
        normalization = torch.sqrt(torch.clamp(variance_x[:, None] * variance_y[None, :], min=1e-14))
        correlation = torch.clamp(covariance / normalization, min=-1.0 + 1e-7, max=1.0 - 1e-7)
        angle = torch.acos(correlation)
        nngp = bias_scale**2 + weight_scale**2 / (2.0 * np.pi) * normalization * (
            torch.sin(angle) + (np.pi - angle) * torch.cos(angle)
        )
        return torch.stack([*rbf, cosine, linear, nngp], dim=0)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        coefficients = self.mixture_coefficients()
        return torch.sum(coefficients[:, None, None] * self.components(x, y), dim=0)

    def regularization(self, outer_l1: float, inner_l2: float) -> torch.Tensor:
        params = self.internal_parameters()
        inner_penalty = torch.sum(torch.stack([value**2 for value in params.values()]))
        return outer_l1 * torch.sum(torch.abs(self.raw_outer_weights)) + inner_l2 * inner_penalty

    def parameter_summary(self) -> dict[str, object]:
        with torch.no_grad():
            normalized = self.normalized_outer_weights().cpu().numpy()
            coefficients = self.mixture_coefficients().cpu().numpy()
            relative_contributions = coefficients / max(np.sum(coefficients), np.finfo(float).eps)
            internal = {key: float(value) for key, value in self.internal_parameters().items()}
        return {
            "raw_outer_weights": self.raw_outer_weights.detach().cpu().numpy().tolist(),
            "normalized_outer_weights": normalized.tolist(),
            "mixture_coefficients": coefficients.tolist(),
            "relative_mixture_contributions": relative_contributions.tolist(),
            "internal_parameters": internal,
        }


class FixedRBFKernel(nn.Module):
    def __init__(self, sigma: float) -> None:
        super().__init__()
        self.register_buffer("sigma", torch.tensor(float(sigma), dtype=torch.float64))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.exp(-pairwise_squared_distance(x, y) / (2.0 * self.sigma**2))


@dataclass
class KernelEDMDModel:
    kernel: nn.Module
    centers: torch.Tensor
    koopman: torch.Tensor
    decoder: torch.Tensor


def regularized_right_solve(cross: torch.Tensor, gram: torch.Tensor, regularization: float) -> torch.Tensor:
    """Compute cross @ (gram + regularization I)^(-1) robustly."""

    identity = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
    matrix = gram + regularization * identity
    try:
        return torch.linalg.solve(matrix.T, cross.T).T
    except RuntimeError:
        return cross @ torch.linalg.pinv(matrix, rtol=1e-10)


def solve_kernel_model(
    kernel: nn.Module,
    centers_x: torch.Tensor,
    centers_y: torch.Tensor,
    koopman_regularization: float,
    modes_regularization: float,
) -> KernelEDMDModel:
    """Analytically update K and the state decoder while fixing the kernel."""

    with torch.no_grad():
        gram = kernel(centers_x, centers_x)
        # Dictionary columns are samples: Psi(Y) = g(Y, centers)^T.
        shifted = kernel(centers_y, centers_x).T
        koopman = regularized_right_solve(shifted, gram, koopman_regularization)
        decoder = regularized_right_solve(centers_x.T, gram, modes_regularization)
    return KernelEDMDModel(kernel, centers_x, koopman, decoder)


def kernel_one_step(model: KernelEDMDModel, states: np.ndarray) -> np.ndarray:
    rows = torch.as_tensor(states.T, dtype=torch.float64)
    with torch.no_grad():
        lifted = model.kernel(rows, model.centers).T
        prediction = model.decoder @ model.koopman @ lifted
    return prediction.cpu().numpy()


def kernel_rollout(model: KernelEDMDModel, initial_state: np.ndarray, steps: int) -> np.ndarray:
    """Recursively apply the state prediction map used by Algorithm 2."""

    state = torch.as_tensor(initial_state[None, :], dtype=torch.float64)
    with torch.no_grad():
        prediction = np.empty((initial_state.size, steps), dtype=float)
        prediction[:, 0] = initial_state
        for step in range(1, steps):
            lifted = model.kernel(state, model.centers).T
            state = (model.decoder @ model.koopman @ lifted).T
            prediction[:, step] = state[0].cpu().numpy()
    return prediction


def kernel_linear_lifted_rollout(
    model: KernelEDMDModel,
    initial_state: np.ndarray,
    steps: int,
) -> np.ndarray:
    """Evolve only in lifted coordinates as a Koopman-invariance diagnostic."""

    initial = torch.as_tensor(initial_state[None, :], dtype=torch.float64)
    with torch.no_grad():
        lifted = model.kernel(initial, model.centers).T
        prediction = np.empty((initial_state.size, steps), dtype=float)
        prediction[:, 0] = initial_state
        for step in range(1, steps):
            lifted = model.koopman @ lifted
            prediction[:, step] = (model.decoder @ lifted)[:, 0].cpu().numpy()
    return prediction


def choose_centers(
    normalized_states: np.ndarray,
    fit_end: int,
    center_count: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    pair_count = fit_end - 1
    if center_count > pair_count:
        raise ValueError("center_count cannot exceed the number of fitted snapshot pairs")
    generator = np.random.default_rng(seed)
    indices = np.sort(generator.choice(pair_count, size=center_count, replace=False))
    centers_x = torch.as_tensor(normalized_states[:, indices].T, dtype=torch.float64)
    centers_y = torch.as_tensor(normalized_states[:, indices + 1].T, dtype=torch.float64)
    return centers_x, centers_y, indices


def koopman_regularization(config: KernelDLConfig, epoch: int) -> float:
    if epoch < config.regularization_switch_epoch:
        return config.koopman_regularization_initial
    return config.koopman_regularization_final


def validation_rollout_error(
    model: KernelEDMDModel,
    normalized_states: np.ndarray,
    start: int,
    end: int,
) -> float:
    prediction = kernel_rollout(model, normalized_states[:, start], end - start)
    return relative_error(normalized_states[:, start:end], prediction)


def train_kernel(
    config: KernelDLConfig,
    normalized_states: np.ndarray,
    fit_end: int,
    epochs: int,
    validation_interval: tuple[int, int] | None,
) -> tuple[LearnedKernelMixture, KernelEDMDModel, LearnedKernelMixture, KernelEDMDModel, list[dict[str, object]]]:
    """Train Algorithm 2 and optionally track a contiguous validation rollout."""

    set_random_seed(config.random_seed)
    kernel = LearnedKernelMixture(config).double()
    initial_kernel = copy.deepcopy(kernel)
    centers_x, centers_y, _ = choose_centers(
        normalized_states,
        fit_end,
        config.center_count,
        config.random_seed,
    )
    initial_model = solve_kernel_model(
        initial_kernel,
        centers_x,
        centers_y,
        config.evaluation_koopman_regularization,
        config.evaluation_modes_regularization,
    )
    optimizer = torch.optim.Adam(kernel.parameters(), lr=config.learning_rate)
    pair_indices = np.arange(fit_end - 1)
    generator = np.random.default_rng(config.random_seed + 1)
    history: list[dict[str, object]] = []

    def record(epoch: int, mean_loss: float) -> None:
        model = solve_kernel_model(
            kernel,
            centers_x,
            centers_y,
            config.evaluation_koopman_regularization,
            config.evaluation_modes_regularization,
        )
        row: dict[str, object] = {
            "epoch": epoch,
            "mean_training_prediction_loss": mean_loss,
            "maximum_eigenvalue_radius": float(torch.max(torch.abs(torch.linalg.eigvals(model.koopman)))),
        }
        if validation_interval is not None:
            row["validation_rollout_relative_error"] = validation_rollout_error(
                model,
                normalized_states,
                validation_interval[0],
                validation_interval[1],
            )
        summary = kernel.parameter_summary()
        for name, coefficient in zip(kernel.component_names, summary["mixture_coefficients"]):
            row[f"coefficient_{name}"] = coefficient
        history.append(row)

    record(0, float("nan"))
    for epoch in range(1, epochs + 1):
        shuffled = generator.permutation(pair_indices)
        losses: list[float] = []
        for batch_indices in np.array_split(shuffled, config.batch_count):
            model = solve_kernel_model(
                kernel,
                centers_x,
                centers_y,
                koopman_regularization(config, epoch),
                config.modes_regularization,
            )
            batch_x = torch.as_tensor(normalized_states[:, batch_indices].T, dtype=torch.float64)
            batch_y = torch.as_tensor(normalized_states[:, batch_indices + 1].T, dtype=torch.float64)
            lifted = kernel(batch_x, centers_x).T
            prediction = (model.decoder.detach() @ model.koopman.detach() @ lifted).T
            prediction_loss = torch.mean((batch_y - prediction) ** 2) / torch.clamp(
                torch.mean(batch_y**2), min=1e-12
            )
            loss = prediction_loss + kernel.regularization(
                config.outer_l1_regularization,
                config.inner_l2_regularization,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(kernel.parameters(), config.gradient_clip)
            optimizer.step()
            losses.append(float(prediction_loss.detach()))
        if epoch % config.validation_interval == 0 or epoch == epochs:
            record(epoch, float(np.mean(losses)))

    final_model = solve_kernel_model(
        kernel,
        centers_x,
        centers_y,
        config.evaluation_koopman_regularization,
        config.evaluation_modes_regularization,
    )
    return kernel, final_model, initial_kernel, initial_model, history


def select_training_epochs(
    config: KernelDLConfig,
    normalized_states: np.ndarray,
    train_end: int,
    validation_end: int,
) -> tuple[int, list[dict[str, object]]]:
    _, _, _, _, history = train_kernel(
        config,
        normalized_states,
        train_end,
        config.max_epochs,
        (train_end, validation_end),
    )
    valid_rows = [
        row
        for row in history
        if "validation_rollout_relative_error" in row
        and np.isfinite(float(row["validation_rollout_relative_error"]))
    ]
    best = min(valid_rows, key=lambda row: float(row["validation_rollout_relative_error"]))
    return int(best["epoch"]), history


def median_rbf_sigma(centers: torch.Tensor) -> float:
    distances = torch.sqrt(pairwise_squared_distance(centers, centers).detach())
    positive = distances[distances > 0]
    return float(torch.median(positive))


def empirical_eigen_residuals(
    model: KernelEDMDModel,
    X: np.ndarray,
    Y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        phi_x = model.kernel(torch.as_tensor(X.T, dtype=torch.float64), model.centers).T.cpu().numpy()
        phi_y = model.kernel(torch.as_tensor(Y.T, dtype=torch.float64), model.centers).T.cpu().numpy()
    koopman = model.koopman.cpu().numpy()
    eigvals, left_vectors = np.linalg.eig(koopman.T)
    residuals = np.empty(len(eigvals), dtype=float)
    for index, eigval in enumerate(eigvals):
        left = left_vectors[:, index].T
        value_x = left @ phi_x
        value_y = left @ phi_y
        residuals[index] = float(
            np.linalg.norm(value_y - eigval * value_x)
            / max(np.linalg.norm(value_x), np.finfo(float).eps)
        )
    return eigvals, residuals


def spectral_rows(
    eigvals: np.ndarray,
    residuals: np.ndarray,
    dt: float,
    reduced_frequency_factor: float,
) -> list[dict[str, float | int]]:
    continuous = np.log(eigvals.astype(complex)) / dt
    frequencies = continuous.imag / (2.0 * np.pi) * reduced_frequency_factor
    rows: list[dict[str, float | int]] = []
    for index, eigval in enumerate(eigvals):
        if eigval.imag <= 1e-9 or abs(eigval) < 0.90 or frequencies[index] > 2.0:
            continue
        rows.append(
            {
                "index_1based": index + 1,
                "reduced_frequency": float(frequencies[index]),
                "growth_rate": float(continuous.real[index]),
                "eigenvalue_radius": float(abs(eigval)),
                "empirical_residual": float(residuals[index]),
            }
        )
    rows.sort(key=lambda row: float(row["reduced_frequency"]))
    return rows


def save_training_history(path: Path, history: list[dict[str, object]]) -> None:
    epochs = [int(row["epoch"]) for row in history]
    training = [float(row["mean_training_prediction_loss"]) for row in history]
    validation = [float(row.get("validation_rollout_relative_error", np.nan)) for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=180)
    axes[0].plot(epochs[1:], training[1:], marker="o", markersize=3)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("normalized prediction loss")
    axes[0].set_title("Algorithm 2 training loss")
    if np.any(np.isfinite(validation)):
        finite = (
            np.isfinite(validation)
            & (np.asarray(validation) > 0.0)
            & (np.asarray(validation) <= 2.0)
        )
        axes[1].plot(np.asarray(epochs)[finite], np.asarray(validation)[finite], marker="o", markersize=3, color="tab:orange")
        axes[1].text(
            0.03,
            0.95,
            "Divergent early values omitted",
            transform=axes[1].transAxes,
            va="top",
            fontsize=9,
        )
        axes[1].set_ylabel("validation rollout error")
        axes[1].set_title("Contiguous validation rollout")
    else:
        radii = [float(row["maximum_eigenvalue_radius"]) for row in history]
        axes[1].plot(epochs, radii, marker="o", markersize=3, color="tab:orange")
        axes[1].axhline(1.0, color="black", linestyle="--", linewidth=0.8)
        axes[1].set_ylabel("maximum eigenvalue radius")
        axes[1].set_title("Lifted linear stability diagnostic")
    axes[1].set_xlabel("epoch")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_kernel_parameters(
    path: Path,
    initial: dict[str, object],
    learned: dict[str, object],
) -> None:
    labels = LearnedKernelMixture.component_names
    initial_coefficients = np.asarray(initial["relative_mixture_contributions"], dtype=float)
    learned_coefficients = np.asarray(learned["relative_mixture_contributions"], dtype=float)
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5), dpi=180)
    width = 0.36
    axes[0].bar(x - width / 2, initial_coefficients, width, label="initial")
    axes[0].bar(x + width / 2, learned_coefficients, width, label="learned")
    axes[0].set_xticks(x, labels, rotation=28, ha="right")
    axes[0].set_ylabel("relative mixture contribution")
    axes[0].set_title("Kernel selection by outer weights")
    axes[0].legend()

    parameter_names = list(initial["internal_parameters"].keys())
    initial_values = [float(initial["internal_parameters"][name]) for name in parameter_names]
    learned_values = [float(learned["internal_parameters"][name]) for name in parameter_names]
    x_parameters = np.arange(len(parameter_names))
    axes[1].bar(x_parameters - width / 2, initial_values, width, label="initial")
    axes[1].bar(x_parameters + width / 2, learned_values, width, label="learned")
    axes[1].set_yscale("log")
    axes[1].set_xticks(x_parameters, parameter_names, rotation=32, ha="right")
    axes[1].set_ylabel("positive internal parameter")
    axes[1].set_title("Learned kernel parameters")
    axes[1].legend()
    for ax in axes:
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_error_comparison(path: Path, metrics: list[dict[str, object]]) -> None:
    metrics = [row for row in metrics if row["model"] != "initial mixture"]
    labels = [str(row["model"]) for row in metrics]
    one_step = [float(row["one_step_state_error"]) for row in metrics]
    rollout = [float(row["rollout_state_error"]) for row in metrics]
    pressure = [float(row["cfd_pressure_fluctuation_error"]) for row in metrics]
    x = np.arange(len(labels))
    width = 0.26
    fig, ax = plt.subplots(figsize=(11.5, 4.8), dpi=180)
    ax.bar(x - width, one_step, width, label="one-step state")
    ax.bar(x, rollout, width, label="300-step state")
    ax.bar(x + width, pressure, width, label="CFD pressure fluctuation")
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_ylabel("relative error")
    ax.set_title("Kernel dictionary learning versus fixed dictionaries")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_state_rollout(
    path: Path,
    time: np.ndarray,
    reference: np.ndarray,
    fixed_rbf: np.ndarray,
    learned: np.ndarray,
) -> None:
    fig, axes = plt.subplots(reference.shape[0], 1, figsize=(10.0, 11.0), dpi=180, sharex=True)
    for index, ax in enumerate(axes):
        ax.plot(time, reference[index], color="black", linewidth=1.15, label="CFD/POD state")
        ax.plot(time, fixed_rbf[index], ":", linewidth=0.9, label="fixed RBF")
        ax.plot(time, learned[index], "-.", linewidth=1.0, label="learned kEDMD-DL")
        ax.set_ylabel(f"z{index + 1}")
        ax.grid(True, alpha=0.25)
    axes[0].legend(ncol=3, fontsize=8)
    axes[-1].set_xlabel("time")
    fig.suptitle("Held-out rollout with a learned kernel dictionary")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_eigenvalue_comparison(
    path: Path,
    initial_eigvals: np.ndarray,
    learned_eigvals: np.ndarray,
    learned_residuals: np.ndarray,
) -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 500)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.7), dpi=180)
    axes[0].plot(np.cos(theta), np.sin(theta), "k--", linewidth=0.9)
    axes[0].scatter(initial_eigvals.real, initial_eigvals.imag, s=25)
    axes[0].set_title("Initial mixed kernel")
    axes[1].plot(np.cos(theta), np.sin(theta), "k--", linewidth=0.9)
    scatter = axes[1].scatter(
        learned_eigvals.real,
        learned_eigvals.imag,
        c=np.log10(np.maximum(learned_residuals, 1e-12)),
        cmap="viridis",
        s=27,
    )
    axes[1].set_title("Learned kEDMD-DL kernel")
    for ax in axes:
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Re(lambda)")
        ax.set_ylabel("Im(lambda)")
        ax.grid(True, alpha=0.25)
    fig.colorbar(scatter, ax=axes[1], shrink=0.82, label="log10 empirical residual")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_harmonic_plot(path: Path, rows: list[dict[str, float | int]]) -> None:
    frequencies = [float(row["reduced_frequency"]) for row in rows]
    radii = [float(row["eigenvalue_radius"]) for row in rows]
    residuals = [float(row["empirical_residual"]) for row in rows]
    fig, ax = plt.subplots(figsize=(7.3, 4.3), dpi=180)
    scatter = ax.scatter(frequencies, radii, c=np.log10(np.maximum(residuals, 1e-12)), cmap="viridis", s=42)
    for harmonic in (0.186, 0.372, 0.558, 0.744, 0.930):
        ax.axvline(harmonic, color="tab:red", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Reduced frequency")
    ax.set_ylabel("Eigenvalue radius")
    ax.set_title("Learned-kernel Koopman spectrum")
    ax.grid(True, alpha=0.25)
    fig.colorbar(scatter, ax=ax, shrink=0.82, label="log10 empirical residual")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_gram_matrices(
    path: Path,
    initial_model: KernelEDMDModel,
    learned_model: KernelEDMDModel,
) -> None:
    with torch.no_grad():
        initial = initial_model.kernel(initial_model.centers, initial_model.centers).cpu().numpy()
        learned = learned_model.kernel(learned_model.centers, learned_model.centers).cpu().numpy()
    vmax = max(float(np.max(np.abs(initial))), float(np.max(np.abs(learned))))
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), dpi=180)
    images = []
    for ax, matrix, title in zip(axes, (initial, learned), ("Initial Gram matrix", "Learned Gram matrix")):
        images.append(ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto"))
        ax.set_title(title)
        ax.set_xlabel("kernel center")
        ax.set_ylabel("kernel center")
    colorbar_axis = fig.add_axes([0.91, 0.18, 0.016, 0.64])
    fig.colorbar(images[-1], cax=colorbar_axis, label="kernel value")
    fig.subplots_adjust(left=0.08, right=0.89, bottom=0.12, top=0.88, wspace=0.25)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_observation_histories(
    path: Path,
    time: np.ndarray,
    true_pressure: np.ndarray,
    pod_pressure: np.ndarray,
    fixed_pressure: np.ndarray,
    learned_pressure: np.ndarray,
    observation_indices: np.ndarray,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(8.8, 6.8), dpi=180, sharex=True)
    for ax, label, index in zip(axes, ("C", "D", "E"), observation_indices):
        ax.plot(time, true_pressure[index], color="black", linewidth=1.0, label="CFD")
        ax.plot(time, pod_pressure[index], "--", linewidth=0.85, label="POD rank 5")
        ax.plot(time, fixed_pressure[index], ":", linewidth=0.9, label="fixed RBF")
        ax.plot(time, learned_pressure[index], "-.", linewidth=0.95, label="learned kEDMD-DL")
        ax.set_ylabel(f"p at {label}")
        ax.grid(True, alpha=0.25)
    axes[0].legend(ncol=4, fontsize=8)
    axes[-1].set_xlabel("time")
    fig.suptitle("Pressure histories with kernel dictionary learning")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_pressure_snapshots(
    path: Path,
    triangulation: mtri.Triangulation,
    time: np.ndarray,
    true_fluctuation: np.ndarray,
    pod_fluctuation: np.ndarray,
    fixed_fluctuation: np.ndarray,
    learned_fluctuation: np.ndarray,
    indices: tuple[int, int],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    datasets = (
        ("CFD", true_fluctuation),
        ("POD rank 5", pod_fluctuation),
        ("fixed RBF", fixed_fluctuation),
        ("learned kEDMD-DL", learned_fluctuation),
    )
    vmax = max(float(np.nanpercentile(np.abs(true_fluctuation[:, index]), 99.0)) for index in indices)
    fig, axes = plt.subplots(2, 4, figsize=(14.0, 5.8), dpi=180, sharex=True, sharey=True)
    last_plot = None
    for row, index in enumerate(indices):
        for column, (label, field) in enumerate(datasets):
            last_plot = axes[row, column].tripcolor(
                triangulation,
                field[:, index],
                shading="gouraud",
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
            )
            axes[row, column].set_xlim(*xlim)
            axes[row, column].set_ylim(*ylim)
            axes[row, column].set_aspect("equal", adjustable="box")
            axes[row, column].set_title(f"{label}, t={time[index]:.3f}")
    for ax in axes[:, 0]:
        ax.set_ylabel("y")
    for ax in axes[-1, :]:
        ax.set_xlabel("x")
    if last_plot is not None:
        colorbar_axis = fig.add_axes([0.91, 0.18, 0.014, 0.62])
        fig.colorbar(last_plot, cax=colorbar_axis, label="pressure fluctuation")
    fig.suptitle("Pressure reconstruction with kEDMD-DL")
    fig.subplots_adjust(left=0.05, right=0.89, bottom=0.10, top=0.88, wspace=0.12, hspace=0.20)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_kedmd_dl_analysis(
    config: KernelDLConfig,
    figures_dir: Path,
    results_dir: Path,
) -> dict[str, object]:
    """Train, evaluate, plot, and save the OAT15A kEDMD-DL experiment."""

    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    reduced_data = np.load(config.pod_dmd_results)
    coefficients = reduced_data["pod_coefficients"][: config.state_rank]
    pod_modes = reduced_data["pod_modes"][:, : config.state_rank]
    mean_pressure = reduced_data["mean_pressure"]
    snapshot_time = reduced_data["snapshot_time"]
    coordinates = reduced_data["coordinates"]
    nodes = reduced_data["nodes"]
    observation_indices = reduced_data["observation_indices"]

    snapshot_count = coefficients.shape[1]
    train_end = int(snapshot_count * config.training_fraction)
    validation_end = int(snapshot_count * (config.training_fraction + config.validation_fraction))
    state_mean = coefficients[:, :train_end].mean(axis=1, keepdims=True)
    state_scale = np.maximum(coefficients[:, :train_end].std(axis=1, keepdims=True), np.finfo(float).eps)
    normalized_states = (coefficients - state_mean) / state_scale

    print("Selecting kEDMD-DL epochs on the validation interval...", flush=True)
    best_epoch, validation_history = select_training_epochs(
        config,
        normalized_states,
        train_end,
        validation_end,
    )
    print(f"Selected epochs: {best_epoch}", flush=True)
    print("Retraining the learned kernel on train + validation data...", flush=True)
    learned_kernel, learned_model, initial_kernel, initial_model, training_history = train_kernel(
        config,
        normalized_states,
        validation_end,
        best_epoch,
        None,
    )

    centers_x, centers_y, center_indices = choose_centers(
        normalized_states,
        validation_end,
        config.center_count,
        config.random_seed,
    )
    sigma = median_rbf_sigma(centers_x)
    fixed_kernel = FixedRBFKernel(sigma).double()
    fixed_model = solve_kernel_model(
        fixed_kernel,
        centers_x,
        centers_y,
        config.evaluation_koopman_regularization,
        config.evaluation_modes_regularization,
    )

    test_reference = normalized_states[:, validation_end:]
    test_x = normalized_states[:, validation_end:-1]
    test_y = normalized_states[:, validation_end + 1 :]
    test_steps = snapshot_count - validation_end
    models = {
        "fixed RBF": fixed_model,
        "initial mixture": initial_model,
        "learned kEDMD-DL": learned_model,
    }
    one_step = {name: kernel_one_step(model, test_x) for name, model in models.items()}
    rollout = {
        name: kernel_rollout(model, normalized_states[:, validation_end], test_steps)
        for name, model in models.items()
    }
    learned_linear_rollout = kernel_linear_lifted_rollout(
        learned_model,
        normalized_states[:, validation_end],
        test_steps,
    )

    pod_config = PodConfig(data_root=config.data_root, data_dir=config.data_dir)
    data_dir = resolve_data_dir(pod_config)
    _, _, full_pressure, _ = load_oat15a_data(data_dir)
    pressure = full_pressure[:, :: config.stride]
    del full_pressure
    true_fluctuation = pressure[:, validation_end:] - mean_pressure[:, None]
    pod_fluctuation = pod_modes @ coefficients[:, validation_end:]
    model_fluctuations = {
        name: pressure_from_states(prediction, state_mean, state_scale, pod_modes)
        for name, prediction in rollout.items()
    }

    metrics: list[dict[str, object]] = []
    for name in models:
        metrics.append(
            {
                "model": name,
                "feature_count": config.center_count,
                "one_step_state_error": relative_error(test_y, one_step[name]),
                "rollout_state_error": relative_error(test_reference, rollout[name]),
                "pod_subspace_pressure_error": relative_error(pod_fluctuation, model_fluctuations[name]),
                "cfd_pressure_fluctuation_error": relative_error(
                    true_fluctuation,
                    model_fluctuations[name],
                ),
            }
        )

    comparison_sources = (
        ("polynomial 3", results_dir.parent / "oat15a_classic_edmd" / "summary.json"),
        ("Q Li EDMD-DL", results_dir.parent / "oat15a_edmd_dl_q_li" / "summary.json"),
    )
    for model_name, summary_path in comparison_sources:
        if not summary_path.exists():
            continue
        source = json.loads(summary_path.read_text(encoding="utf-8"))
        source_metrics = source.get("model_comparison", source.get("degree_test_metrics", []))
        if model_name == "polynomial 3":
            row = next(item for item in source_metrics if int(item.get("degree", -1)) == 3)
            metrics.append(
                {
                    "model": model_name,
                    "feature_count": row["feature_count"],
                    "one_step_state_error": row["test_one_step_relative_error"],
                    "rollout_state_error": row["test_rollout_relative_error"],
                    "pod_subspace_pressure_error": row["pod_subspace_pressure_relative_error"],
                    "cfd_pressure_fluctuation_error": row["cfd_pressure_fluctuation_relative_error"],
                }
            )
        else:
            row = next(item for item in source_metrics if item.get("model") == model_name)
            metrics.append(dict(row))

    learned_eigvals, learned_residuals = empirical_eigen_residuals(learned_model, test_x, test_y)
    initial_eigvals = np.linalg.eigvals(initial_model.koopman.cpu().numpy())
    learned_spectrum = spectral_rows(
        learned_eigvals,
        learned_residuals,
        config.dt,
        config.reduced_frequency_factor,
    )
    initial_parameters = initial_kernel.parameter_summary()
    learned_parameters = learned_kernel.parameter_summary()
    test_time = snapshot_time[validation_end:]

    save_training_history(figures_dir / "kedmd_dl_validation_history.png", validation_history)
    save_training_history(figures_dir / "kedmd_dl_final_training_history.png", training_history)
    save_kernel_parameters(
        figures_dir / "kedmd_dl_kernel_parameters.png",
        initial_parameters,
        learned_parameters,
    )
    save_error_comparison(figures_dir / "kedmd_dl_error_comparison.png", metrics)
    save_state_rollout(
        figures_dir / "kedmd_dl_state_rollout.png",
        test_time,
        test_reference,
        rollout["fixed RBF"],
        rollout["learned kEDMD-DL"],
    )
    save_eigenvalue_comparison(
        figures_dir / "kedmd_dl_eigenvalue_comparison.png",
        initial_eigvals,
        learned_eigvals,
        learned_residuals,
    )
    save_harmonic_plot(figures_dir / "kedmd_dl_harmonic_spectrum.png", learned_spectrum)
    save_gram_matrices(figures_dir / "kedmd_dl_gram_matrices.png", initial_model, learned_model)

    triangulation = make_triangulation(coordinates, nodes)
    true_pressure = pressure[:, validation_end:]
    pod_pressure = mean_pressure[:, None] + pod_fluctuation
    fixed_pressure = mean_pressure[:, None] + model_fluctuations["fixed RBF"]
    learned_pressure = mean_pressure[:, None] + model_fluctuations["learned kEDMD-DL"]
    save_observation_histories(
        figures_dir / "kedmd_dl_observation_histories.png",
        test_time,
        true_pressure,
        pod_pressure,
        fixed_pressure,
        learned_pressure,
        observation_indices,
    )
    comparison_indices = (0, min(150, test_steps - 1))
    save_pressure_snapshots(
        figures_dir / "kedmd_dl_pressure_snapshot_comparison.png",
        triangulation,
        test_time,
        true_fluctuation,
        pod_fluctuation,
        model_fluctuations["fixed RBF"],
        model_fluctuations["learned kEDMD-DL"],
        comparison_indices,
        config.zoom_xlim,
        config.zoom_ylim,
    )

    torch.save(
        {
            "kernel_state_dict": learned_kernel.state_dict(),
            "centers": learned_model.centers.cpu(),
            "koopman_matrix": learned_model.koopman.cpu(),
            "decoder": learned_model.decoder.cpu(),
            "state_mean": state_mean,
            "state_scale": state_scale,
            "config": asdict(config),
        },
        results_dir / "oat15a_kedmd_dl_model.pt",
    )
    np.savez_compressed(
        results_dir / "oat15a_kedmd_dl_results.npz",
        snapshot_time=snapshot_time,
        normalized_states=normalized_states,
        test_reference=test_reference,
        fixed_rbf_rollout=rollout["fixed RBF"],
        initial_mixture_rollout=rollout["initial mixture"],
        learned_rollout=rollout["learned kEDMD-DL"],
        learned_linear_lifted_rollout=learned_linear_rollout,
        learned_eigvals=learned_eigvals,
        learned_residuals=learned_residuals,
        centers=learned_model.centers.cpu().numpy(),
        center_indices=center_indices,
        state_mean=state_mean,
        state_scale=state_scale,
        pod_modes=pod_modes,
        mean_pressure=mean_pressure,
        observation_indices=observation_indices,
    )
    write_csv(results_dir / "validation_history.csv", validation_history)
    write_csv(results_dir / "training_history.csv", training_history)
    write_csv(results_dir / "model_comparison.csv", metrics)
    write_csv(results_dir / "learned_spectrum.csv", learned_spectrum)

    learned_metrics = next(row for row in metrics if row["model"] == "learned kEDMD-DL")
    fixed_metrics = next(row for row in metrics if row["model"] == "fixed RBF")
    summary: dict[str, object] = {
        "reference": "Bolager et al. 2026, Dictionary Learning for Kernel EDMD, Algorithm 2",
        "method": "simplified kEDMD with learned weighted kernel mixture and prediction loss",
        "primary_rollout_definition": "recursive state prediction z_next = Cp K g(z, centers)",
        "state_definition": f"first {config.state_rank} POD pressure coefficients",
        "kernel_components": list(LearnedKernelMixture.component_names),
        "center_count": config.center_count,
        "selected_training_epochs": best_epoch,
        "train_snapshot_count": train_end,
        "validation_snapshot_count": validation_end - train_end,
        "test_snapshot_count": snapshot_count - validation_end,
        "fixed_rbf_sigma": sigma,
        "initial_kernel_parameters": initial_parameters,
        "learned_kernel_parameters": learned_parameters,
        "model_comparison": metrics,
        "learned_spectrum": learned_spectrum,
        "learned_rollout_error": learned_metrics["rollout_state_error"],
        "learned_linear_lifted_rollout_error": relative_error(
            test_reference,
            learned_linear_rollout,
        ),
        "fixed_rbf_rollout_error": fixed_metrics["rollout_state_error"],
        "rollout_error_reduction_from_fixed_rbf": 1.0
        - float(learned_metrics["rollout_state_error"]) / float(fixed_metrics["rollout_state_error"]),
        "pod_rank5_cfd_pressure_fluctuation_error": relative_error(true_fluctuation, pod_fluctuation),
        "learned_pressure_error": learned_metrics["cfd_pressure_fluctuation_error"],
        "figures_dir": str(figures_dir),
        "model_file": str(results_dir / "oat15a_kedmd_dl_model.pt"),
        "results_file": str(results_dir / "oat15a_kedmd_dl_results.npz"),
    }
    (results_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (results_dir / "config.json").write_text(
        json.dumps(asdict(config), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved figures to: {figures_dir}", flush=True)
    print(f"Saved results to: {results_dir}", flush=True)
    return summary
