"""Qianxiao Li et al. (2017) EDMD with dictionary learning for OAT15A.

The pressure field is represented by five leading POD coefficients. The EDMD
dictionary contains fixed constant/coordinate observables and trainable neural
observables from a three-hidden-layer tanh network. Training alternates between
an analytic Koopman update and a gradient update of the dictionary.
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

from edmd import PolynomialDictionary, edmd_one_step, edmd_rollout, fit_edmd
from oat15a_pod import PodConfig, load_oat15a_data, make_triangulation, resolve_data_dir


@dataclass
class EDMDLearningConfig:
    """Configuration for Q Li-style explicit neural dictionary learning."""

    data_root: str = str(Path(__file__).resolve().parents[1] / "data" / "oat15a")
    data_dir: str | None = None
    pod_dmd_results: str = str(
        Path(__file__).resolve().parents[1]
        / "results" / "oat15a_pod_dmd_stride2" / "oat15a_stride2_pod_dmd_results.npz"
    )
    stride: int = 2
    state_rank: int = 5
    training_fraction: float = 0.60
    validation_fraction: float = 0.10
    trainable_observables: int = 50
    hidden_width: int = 64
    hidden_layers: int = 3
    learning_rate: float = 5e-4
    koopman_regularization: float = 1e-6
    max_epochs: int = 4000
    validation_interval: int = 100
    gradient_clip: float = 10.0
    random_seed: int = 8
    classic_degrees: tuple[int, ...] = (2, 3)
    reduced_frequency_factor: float = float(np.pi / 2.0)
    dt: float = 0.07295943052
    zoom_xlim: tuple[float, float] = (-0.2, 1.2)
    zoom_ylim: tuple[float, float] = (-0.4, 0.4)


class NeuralDictionary(nn.Module):
    """Fixed identity observables plus a three-layer tanh neural dictionary."""

    def __init__(
        self,
        state_dimension: int,
        trainable_observables: int,
        hidden_width: int,
        hidden_layers: int,
    ) -> None:
        super().__init__()
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be positive")

        layers: list[nn.Module] = []
        input_dimension = state_dimension
        for _ in range(hidden_layers):
            layers.append(nn.Linear(input_dimension, hidden_width))
            layers.append(nn.Tanh())
            input_dimension = hidden_width
        layers.append(nn.Linear(input_dimension, trainable_observables))
        self.network = nn.Sequential(*layers)
        self.state_dimension = state_dimension
        self.trainable_observables = trainable_observables

    @property
    def feature_count(self) -> int:
        return 1 + self.state_dimension + self.trainable_observables

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """Evaluate ``[1, x, neural_observables(x)]`` for row-wise states."""

        bias = torch.ones((states.shape[0], 1), dtype=states.dtype, device=states.device)
        return torch.cat([bias, states, self.network(states)], dim=1)


def set_random_seed(seed: int) -> None:
    """Set NumPy and Torch seeds for a reproducible learned dictionary."""

    np.random.seed(seed)
    torch.manual_seed(seed)


def relative_error(reference: np.ndarray, approximation: np.ndarray) -> float:
    """Return relative Frobenius error."""

    denominator = np.linalg.norm(reference)
    if denominator == 0.0:
        return float(np.linalg.norm(reference - approximation))
    return float(np.linalg.norm(reference - approximation) / denominator)


def solve_koopman(
    dictionary: NeuralDictionary,
    X: torch.Tensor,
    Y: torch.Tensor,
    regularization: float,
) -> torch.Tensor:
    """Solve the regularized EDMD problem while holding the dictionary fixed."""

    with torch.no_grad():
        phi_x = dictionary(X).T
        phi_y = dictionary(Y).T
        gram = phi_x @ phi_x.T
        cross = phi_y @ phi_x.T
        identity = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
        # Column convention: Phi_Y ~= K Phi_X.
        koopman = torch.linalg.solve(gram + regularization * identity, cross.T).T
    return koopman


def neural_one_step(
    dictionary: NeuralDictionary,
    koopman: torch.Tensor,
    states: np.ndarray,
) -> np.ndarray:
    """Predict one step by lifting, applying K, and reading coordinate observables."""

    tensor = torch.as_tensor(states.T, dtype=torch.float64)
    with torch.no_grad():
        phi = dictionary(tensor).T
        phi_next = koopman @ phi
        prediction = phi_next[1 : 1 + dictionary.state_dimension]
    return prediction.cpu().numpy()


def neural_rollout(
    dictionary: NeuralDictionary,
    koopman: torch.Tensor,
    initial_state: np.ndarray,
    steps: int,
) -> np.ndarray:
    """Evolve the learned observables linearly for multiple steps."""

    initial = torch.as_tensor(initial_state[None, :], dtype=torch.float64)
    with torch.no_grad():
        phi = dictionary(initial).T
        prediction = np.empty((dictionary.state_dimension, steps), dtype=float)
        for step in range(steps):
            prediction[:, step] = phi[1 : 1 + dictionary.state_dimension, 0].cpu().numpy()
            phi = koopman @ phi
    return prediction


def dictionary_closure_loss(
    dictionary: NeuralDictionary,
    koopman: torch.Tensor,
    X: torch.Tensor,
    Y: torch.Tensor,
) -> torch.Tensor:
    """Return the normalized EDMD dictionary-invariance residual."""

    phi_x = dictionary(X).T
    phi_y = dictionary(Y).T
    return torch.mean((phi_y - koopman.detach() @ phi_x) ** 2)


def validation_rollout_error(
    dictionary: NeuralDictionary,
    koopman: torch.Tensor,
    normalized_states: np.ndarray,
    start: int,
    end: int,
) -> float:
    """Evaluate a multi-step validation rollout."""

    prediction = neural_rollout(
        dictionary,
        koopman,
        normalized_states[:, start],
        end - start,
    )
    return relative_error(normalized_states[:, start:end], prediction)


def select_training_epochs(
    config: EDMDLearningConfig,
    normalized_states: np.ndarray,
    train_end: int,
    validation_end: int,
) -> tuple[int, list[dict[str, float | int]]]:
    """Use the validation interval to select the number of alternating updates."""

    set_random_seed(config.random_seed)
    dictionary = NeuralDictionary(
        config.state_rank,
        config.trainable_observables,
        config.hidden_width,
        config.hidden_layers,
    ).double()
    optimizer = torch.optim.Adam(dictionary.parameters(), lr=config.learning_rate)
    X = torch.as_tensor(normalized_states[:, : train_end - 1].T, dtype=torch.float64)
    Y = torch.as_tensor(normalized_states[:, 1:train_end].T, dtype=torch.float64)

    history: list[dict[str, float | int]] = []
    best_epoch = 0
    best_error = float("inf")
    for epoch in range(config.max_epochs + 1):
        if epoch % config.validation_interval == 0 or epoch == config.max_epochs:
            koopman = solve_koopman(
                dictionary,
                X,
                Y,
                config.koopman_regularization,
            )
            error = validation_rollout_error(
                dictionary,
                koopman,
                normalized_states,
                train_end,
                validation_end,
            )
            closure = float(dictionary_closure_loss(dictionary, koopman, X, Y).detach())
            history.append(
                {
                    "epoch": epoch,
                    "validation_rollout_relative_error": error,
                    "training_dictionary_loss": closure,
                }
            )
            if error < best_error:
                best_error = error
                best_epoch = epoch

        if epoch == config.max_epochs:
            break
        koopman = solve_koopman(
            dictionary,
            X,
            Y,
            config.koopman_regularization,
        )
        loss = dictionary_closure_loss(dictionary, koopman, X, Y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dictionary.parameters(), config.gradient_clip)
        optimizer.step()
    return best_epoch, history


def train_final_dictionary(
    config: EDMDLearningConfig,
    normalized_states: np.ndarray,
    fit_end: int,
    epochs: int,
) -> tuple[
    NeuralDictionary,
    torch.Tensor,
    NeuralDictionary,
    torch.Tensor,
    list[dict[str, float | int]],
]:
    """Reset and train on the combined training and validation interval."""

    set_random_seed(config.random_seed)
    dictionary = NeuralDictionary(
        config.state_rank,
        config.trainable_observables,
        config.hidden_width,
        config.hidden_layers,
    ).double()
    initial_dictionary = copy.deepcopy(dictionary)
    optimizer = torch.optim.Adam(dictionary.parameters(), lr=config.learning_rate)
    X = torch.as_tensor(normalized_states[:, : fit_end - 1].T, dtype=torch.float64)
    Y = torch.as_tensor(normalized_states[:, 1:fit_end].T, dtype=torch.float64)
    initial_koopman = solve_koopman(
        initial_dictionary,
        X,
        Y,
        config.koopman_regularization,
    )

    history: list[dict[str, float | int]] = []
    record_interval = max(1, min(50, epochs // 20 if epochs else 1))
    for epoch in range(epochs):
        koopman = solve_koopman(
            dictionary,
            X,
            Y,
            config.koopman_regularization,
        )
        loss = dictionary_closure_loss(dictionary, koopman, X, Y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dictionary.parameters(), config.gradient_clip)
        optimizer.step()
        if epoch % record_interval == 0 or epoch == epochs - 1:
            history.append({"epoch": epoch + 1, "training_dictionary_loss": float(loss.detach())})

    final_koopman = solve_koopman(
        dictionary,
        X,
        Y,
        config.koopman_regularization,
    )
    return dictionary, final_koopman, initial_dictionary, initial_koopman, history


def empirical_eigen_residuals(
    dictionary: NeuralDictionary,
    koopman: torch.Tensor,
    X: np.ndarray,
    Y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute held-out eigenfunction residuals for the learned dictionary."""

    with torch.no_grad():
        phi_x = dictionary(torch.as_tensor(X.T, dtype=torch.float64)).T.cpu().numpy()
        phi_y = dictionary(torch.as_tensor(Y.T, dtype=torch.float64)).T.cpu().numpy()
    koopman_numpy = koopman.cpu().numpy()
    eigvals, left_vectors = np.linalg.eig(koopman_numpy.T)
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
    """Extract positive-frequency near-unit learned Koopman eigenpairs."""

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


def pressure_from_states(
    normalized_states: np.ndarray,
    state_mean: np.ndarray,
    state_scale: np.ndarray,
    pod_modes: np.ndarray,
) -> np.ndarray:
    """Map normalized POD coordinates back to pressure fluctuations."""

    coefficients = normalized_states * state_scale + state_mean
    return pod_modes @ coefficients


def save_validation_history(path: Path, rows: list[dict[str, float | int]]) -> None:
    """Plot validation rollout error and dictionary loss during epoch selection."""

    epochs = [int(row["epoch"]) for row in rows]
    errors = [float(row["validation_rollout_relative_error"]) for row in rows]
    losses = [float(row["training_dictionary_loss"]) for row in rows]
    fig, ax1 = plt.subplots(figsize=(7.2, 4.2), dpi=180)
    ax1.semilogy(epochs, errors, "o-", label="validation rollout error")
    ax1.set_xlabel("Alternating-training epoch")
    ax1.set_ylabel("Validation relative error")
    ax1.grid(True, which="both", alpha=0.3)
    ax2 = ax1.twinx()
    ax2.semilogy(epochs, losses, "s--", color="tab:red", label="dictionary loss")
    ax2.set_ylabel("Dictionary closure loss")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines])
    ax1.set_title("Q Li EDMD-DL validation history")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_training_history(path: Path, rows: list[dict[str, float | int]]) -> None:
    """Plot final alternating-training dictionary loss."""

    fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=180)
    ax.semilogy(
        [int(row["epoch"]) for row in rows],
        [float(row["training_dictionary_loss"]) for row in rows],
        linewidth=1.2,
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Dictionary closure loss")
    ax.set_title("Final Q Li EDMD-DL training")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_error_comparison(path: Path, metrics: list[dict[str, float | int | str]]) -> None:
    """Compare random, polynomial, and learned dictionaries."""

    labels = [str(row["model"]) for row in metrics]
    one_step = [float(row["one_step_state_error"]) for row in metrics]
    rollout = [float(row["rollout_state_error"]) for row in metrics]
    pressure = [float(row["cfd_pressure_fluctuation_error"]) for row in metrics]
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8.6, 4.4), dpi=180)
    ax.bar(x - width, one_step, width, label="one-step state")
    ax.bar(x, rollout, width, label="300-step state")
    ax.bar(x + width, pressure, width, label="CFD pressure fluctuation")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Relative error")
    ax.set_title("Fixed versus learned EDMD dictionaries")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_state_rollout(
    path: Path,
    time: np.ndarray,
    reference: np.ndarray,
    random_prediction: np.ndarray,
    polynomial_prediction: np.ndarray,
    learned_prediction: np.ndarray,
) -> None:
    """Plot test POD-state trajectories for the main models."""

    fig, axes = plt.subplots(reference.shape[0], 1, figsize=(9.0, 1.8 * reference.shape[0]), dpi=180, sharex=True)
    for index, ax in enumerate(np.atleast_1d(axes)):
        ax.plot(time, reference[index], color="black", linewidth=1.1, label="CFD/POD state")
        ax.plot(time, random_prediction[index], color="0.65", linewidth=0.8, label="random dictionary")
        ax.plot(time, polynomial_prediction[index], "--", linewidth=0.9, label="degree 3 EDMD")
        ax.plot(time, learned_prediction[index], "-.", linewidth=0.95, label="Q Li EDMD-DL")
        ax.set_ylabel(f"z{index + 1}")
        ax.grid(True, alpha=0.25)
    np.atleast_1d(axes)[0].legend(ncol=4, fontsize=8)
    np.atleast_1d(axes)[-1].set_xlabel("time")
    fig.suptitle("Held-out rollout with a learned EDMD dictionary")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_eigenvalue_comparison(
    path: Path,
    polynomial_eigvals: np.ndarray,
    learned_eigvals: np.ndarray,
    learned_residuals: np.ndarray,
) -> None:
    """Compare polynomial and learned Koopman spectra."""

    theta = np.linspace(0.0, 2.0 * np.pi, 500)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.7), dpi=180)
    axes[0].plot(np.cos(theta), np.sin(theta), "k--", linewidth=0.9)
    axes[0].scatter(polynomial_eigvals.real, polynomial_eigvals.imag, s=28)
    axes[0].set_title("Degree 3 polynomial EDMD")
    axes[1].plot(np.cos(theta), np.sin(theta), "k--", linewidth=0.9)
    scatter = axes[1].scatter(
        learned_eigvals.real,
        learned_eigvals.imag,
        c=np.log10(np.maximum(learned_residuals, 1e-12)),
        cmap="viridis",
        s=30,
    )
    axes[1].set_title("Q Li EDMD-DL")
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
    """Plot the learned near-unit spectrum against buffet harmonics."""

    frequencies = [float(row["reduced_frequency"]) for row in rows]
    radii = [float(row["eigenvalue_radius"]) for row in rows]
    residuals = [float(row["empirical_residual"]) for row in rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=180)
    scatter = ax.scatter(frequencies, radii, c=np.log10(np.maximum(residuals, 1e-12)), cmap="viridis", s=42)
    for harmonic in (0.186, 0.372, 0.558, 0.744, 0.930):
        ax.axvline(harmonic, color="tab:red", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Reduced frequency")
    ax.set_ylabel("Eigenvalue radius")
    ax.set_title("Learned-dictionary Koopman spectrum")
    ax.grid(True, alpha=0.25)
    fig.colorbar(scatter, ax=ax, shrink=0.82, label="log10 empirical residual")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_learned_observables(
    path: Path,
    time: np.ndarray,
    dictionary: NeuralDictionary,
    normalized_states: np.ndarray,
    count: int = 10,
) -> None:
    """Display several learned observables over the full trajectory."""

    with torch.no_grad():
        features = dictionary(
            torch.as_tensor(normalized_states.T, dtype=torch.float64)
        ).cpu().numpy()
    learned = features[:, 1 + dictionary.state_dimension : 1 + dictionary.state_dimension + count].T
    learned = (learned - learned.mean(axis=1, keepdims=True)) / np.maximum(
        learned.std(axis=1, keepdims=True),
        np.finfo(float).eps,
    )
    fig, ax = plt.subplots(figsize=(9.0, 4.5), dpi=180)
    image = ax.imshow(
        learned,
        aspect="auto",
        origin="lower",
        extent=(time[0], time[-1], 1, count),
        cmap="RdBu_r",
        vmin=-2.5,
        vmax=2.5,
    )
    ax.set_xlabel("time")
    ax.set_ylabel("learned observable index")
    ax.set_title("First learned neural dictionary observables")
    fig.colorbar(image, ax=ax, shrink=0.85, label="standardized observable")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_observation_histories(
    path: Path,
    time: np.ndarray,
    true_pressure: np.ndarray,
    pod_pressure: np.ndarray,
    polynomial_pressure: np.ndarray,
    learned_pressure: np.ndarray,
    observation_indices: np.ndarray,
) -> None:
    """Compare pressure histories at the three shock-region points."""

    fig, axes = plt.subplots(3, 1, figsize=(8.8, 6.8), dpi=180, sharex=True)
    for ax, label, index in zip(axes, ("C", "D", "E"), observation_indices):
        ax.plot(time, true_pressure[index], color="black", linewidth=1.0, label="CFD")
        ax.plot(time, pod_pressure[index], "--", linewidth=0.85, label="POD rank 5")
        ax.plot(time, polynomial_pressure[index], ":", linewidth=0.9, label="degree 3 EDMD")
        ax.plot(time, learned_pressure[index], "-.", linewidth=0.95, label="Q Li EDMD-DL")
        ax.set_ylabel(f"p at {label}")
        ax.grid(True, alpha=0.25)
    axes[0].legend(ncol=4, fontsize=8)
    axes[-1].set_xlabel("time")
    fig.suptitle("Pressure histories with a learned EDMD dictionary")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_pressure_snapshots(
    path: Path,
    triangulation: mtri.Triangulation,
    time: np.ndarray,
    true_fluctuation: np.ndarray,
    pod_fluctuation: np.ndarray,
    polynomial_fluctuation: np.ndarray,
    learned_fluctuation: np.ndarray,
    indices: tuple[int, int],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    """Compare learned-dictionary pressure fields at two test instants."""

    datasets = (
        ("CFD", true_fluctuation),
        ("POD rank 5", pod_fluctuation),
        ("degree 3 EDMD", polynomial_fluctuation),
        ("Q Li EDMD-DL", learned_fluctuation),
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
    fig.suptitle("Pressure reconstruction with Q Li EDMD-DL")
    fig.subplots_adjust(left=0.05, right=0.89, bottom=0.10, top=0.88, wspace=0.12, hspace=0.20)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write rows to UTF-8 CSV."""

    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_edmd_dl_analysis(
    config: EDMDLearningConfig,
    figures_dir: Path,
    results_dir: Path,
) -> dict[str, object]:
    """Train and evaluate Q Li-style EDMD dictionary learning on OAT15A."""

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
    state_scale = np.maximum(
        coefficients[:, :train_end].std(axis=1, keepdims=True),
        np.finfo(float).eps,
    )
    normalized_states = (coefficients - state_mean) / state_scale

    print("Selecting EDMD-DL training epochs on the validation interval...", flush=True)
    best_epoch, validation_history = select_training_epochs(
        config,
        normalized_states,
        train_end,
        validation_end,
    )
    print(f"Selected epochs: {best_epoch}", flush=True)
    print("Retraining the learned dictionary on train + validation data...", flush=True)
    dictionary, koopman, initial_dictionary, initial_koopman, training_history = train_final_dictionary(
        config,
        normalized_states,
        validation_end,
        best_epoch,
    )

    test_reference = normalized_states[:, validation_end:]
    test_x = normalized_states[:, validation_end:-1]
    test_y = normalized_states[:, validation_end + 1 :]
    test_steps = snapshot_count - validation_end
    initial_one_step = neural_one_step(initial_dictionary, initial_koopman, test_x)
    learned_one_step = neural_one_step(dictionary, koopman, test_x)
    initial_rollout = neural_rollout(
        initial_dictionary,
        initial_koopman,
        normalized_states[:, validation_end],
        test_steps,
    )
    learned_rollout = neural_rollout(
        dictionary,
        koopman,
        normalized_states[:, validation_end],
        test_steps,
    )

    polynomial_models: dict[int, object] = {}
    polynomial_predictions: dict[int, np.ndarray] = {}
    polynomial_one_step: dict[int, np.ndarray] = {}
    for degree in config.classic_degrees:
        polynomial_dictionary = PolynomialDictionary(config.state_rank, degree, include_bias=True)
        model = fit_edmd(
            normalized_states[:, : validation_end - 1],
            normalized_states[:, 1:validation_end],
            polynomial_dictionary,
            rcond=1e-8,
        )
        polynomial_models[degree] = model
        polynomial_predictions[degree] = edmd_rollout(
            model,
            normalized_states[:, validation_end],
            test_steps,
        )
        polynomial_one_step[degree] = edmd_one_step(model, test_x)

    pod_config = PodConfig(data_root=config.data_root, data_dir=config.data_dir)
    data_dir = resolve_data_dir(pod_config)
    _, _, full_pressure, _ = load_oat15a_data(data_dir)
    pressure = full_pressure[:, :: config.stride]
    del full_pressure
    true_fluctuation = pressure[:, validation_end:] - mean_pressure[:, None]
    pod_fluctuation = pod_modes @ coefficients[:, validation_end:]
    initial_fluctuation = pressure_from_states(
        initial_rollout,
        state_mean,
        state_scale,
        pod_modes,
    )
    learned_fluctuation = pressure_from_states(
        learned_rollout,
        state_mean,
        state_scale,
        pod_modes,
    )
    polynomial_fluctuations = {
        degree: pressure_from_states(prediction, state_mean, state_scale, pod_modes)
        for degree, prediction in polynomial_predictions.items()
    }

    metrics: list[dict[str, float | int | str]] = [
        {
            "model": "random neural",
            "feature_count": initial_dictionary.feature_count,
            "one_step_state_error": relative_error(test_y, initial_one_step),
            "rollout_state_error": relative_error(test_reference, initial_rollout),
            "pod_subspace_pressure_error": relative_error(pod_fluctuation, initial_fluctuation),
            "cfd_pressure_fluctuation_error": relative_error(true_fluctuation, initial_fluctuation),
        }
    ]
    for degree in config.classic_degrees:
        metrics.append(
            {
                "model": f"polynomial {degree}",
                "feature_count": polynomial_models[degree].dictionary.n_features,
                "one_step_state_error": relative_error(test_y, polynomial_one_step[degree]),
                "rollout_state_error": relative_error(test_reference, polynomial_predictions[degree]),
                "pod_subspace_pressure_error": relative_error(
                    pod_fluctuation,
                    polynomial_fluctuations[degree],
                ),
                "cfd_pressure_fluctuation_error": relative_error(
                    true_fluctuation,
                    polynomial_fluctuations[degree],
                ),
            }
        )
    metrics.append(
        {
            "model": "Q Li EDMD-DL",
            "feature_count": dictionary.feature_count,
            "one_step_state_error": relative_error(test_y, learned_one_step),
            "rollout_state_error": relative_error(test_reference, learned_rollout),
            "pod_subspace_pressure_error": relative_error(pod_fluctuation, learned_fluctuation),
            "cfd_pressure_fluctuation_error": relative_error(true_fluctuation, learned_fluctuation),
        }
    )

    learned_eigvals, learned_residuals = empirical_eigen_residuals(
        dictionary,
        koopman,
        test_x,
        test_y,
    )
    learned_spectrum = spectral_rows(
        learned_eigvals,
        learned_residuals,
        config.dt,
        config.reduced_frequency_factor,
    )
    polynomial_eigvals = np.linalg.eigvals(polynomial_models[3].koopman_matrix)

    test_time = snapshot_time[validation_end:]
    triangulation = make_triangulation(coordinates, nodes)
    save_validation_history(figures_dir / "edmd_dl_validation_history.png", validation_history)
    save_training_history(figures_dir / "edmd_dl_training_history.png", training_history)
    save_error_comparison(figures_dir / "edmd_dl_error_comparison.png", metrics)
    save_state_rollout(
        figures_dir / "edmd_dl_state_rollout.png",
        test_time,
        test_reference,
        initial_rollout,
        polynomial_predictions[3],
        learned_rollout,
    )
    save_eigenvalue_comparison(
        figures_dir / "edmd_dl_eigenvalue_comparison.png",
        polynomial_eigvals,
        learned_eigvals,
        learned_residuals,
    )
    save_harmonic_plot(figures_dir / "edmd_dl_harmonic_spectrum.png", learned_spectrum)
    save_learned_observables(
        figures_dir / "edmd_dl_learned_observables.png",
        snapshot_time,
        dictionary,
        normalized_states,
    )

    true_pressure = pressure[:, validation_end:]
    pod_pressure = mean_pressure[:, None] + pod_fluctuation
    polynomial_pressure = mean_pressure[:, None] + polynomial_fluctuations[3]
    learned_pressure = mean_pressure[:, None] + learned_fluctuation
    save_observation_histories(
        figures_dir / "edmd_dl_observation_histories.png",
        test_time,
        true_pressure,
        pod_pressure,
        polynomial_pressure,
        learned_pressure,
        observation_indices,
    )
    comparison_indices = (0, min(150, test_steps - 1))
    save_pressure_snapshots(
        figures_dir / "edmd_dl_pressure_snapshot_comparison.png",
        triangulation,
        test_time,
        true_fluctuation,
        pod_fluctuation,
        polynomial_fluctuations[3],
        learned_fluctuation,
        comparison_indices,
        config.zoom_xlim,
        config.zoom_ylim,
    )

    torch.save(
        {
            "state_dict": dictionary.state_dict(),
            "koopman_matrix": koopman.cpu(),
            "config": asdict(config),
            "state_mean": state_mean,
            "state_scale": state_scale,
        },
        results_dir / "oat15a_edmd_dl_model.pt",
    )
    np.savez_compressed(
        results_dir / "oat15a_edmd_dl_results.npz",
        snapshot_time=snapshot_time,
        normalized_states=normalized_states,
        test_reference=test_reference,
        initial_rollout=initial_rollout,
        polynomial_degree3_rollout=polynomial_predictions[3],
        learned_rollout=learned_rollout,
        learned_eigvals=learned_eigvals,
        learned_residuals=learned_residuals,
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

    learned_metrics = next(row for row in metrics if row["model"] == "Q Li EDMD-DL")
    polynomial_metrics = next(row for row in metrics if row["model"] == "polynomial 3")
    pod_rank5_pressure_error = relative_error(true_fluctuation, pod_fluctuation)
    summary: dict[str, object] = {
        "reference": "Li et al. 2017, Extended dynamic mode decomposition with dictionary learning",
        "method": "alternating EDMD-DL with a three-hidden-layer tanh neural dictionary",
        "state_definition": f"first {config.state_rank} POD pressure coefficients",
        "fixed_observables": ["1"] + [f"z{i + 1}" for i in range(config.state_rank)],
        "trainable_observable_count": config.trainable_observables,
        "total_feature_count": dictionary.feature_count,
        "selected_training_epochs": best_epoch,
        "train_snapshot_count": train_end,
        "validation_snapshot_count": validation_end - train_end,
        "test_snapshot_count": snapshot_count - validation_end,
        "model_comparison": metrics,
        "learned_spectrum": learned_spectrum,
        "learned_rollout_error": learned_metrics["rollout_state_error"],
        "polynomial_degree3_rollout_error": polynomial_metrics["rollout_state_error"],
        "equal_feature_rollout_error_reduction": 1.0
        - float(learned_metrics["rollout_state_error"])
        / float(polynomial_metrics["rollout_state_error"]),
        "pod_rank5_cfd_pressure_fluctuation_error": pod_rank5_pressure_error,
        "learned_pressure_error": learned_metrics["cfd_pressure_fluctuation_error"],
        "polynomial_degree3_pressure_error": polynomial_metrics[
            "cfd_pressure_fluctuation_error"
        ],
        "figures_dir": str(figures_dir),
        "model_file": str(results_dir / "oat15a_edmd_dl_model.pt"),
        "results_file": str(results_dir / "oat15a_edmd_dl_results.npz"),
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
