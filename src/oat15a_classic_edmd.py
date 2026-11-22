"""Classical polynomial EDMD analysis for OAT15A transonic buffet data.

The 8100-dimensional pressure field is first represented by leading POD
coefficients. Classical EDMD then uses an explicit polynomial dictionary of
those reduced states; no learned or kernel dictionary is used.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

from edmd import PolynomialDictionary, edmd_one_step, edmd_rollout, fit_edmd
from oat15a_pod import PodConfig, load_oat15a_data, make_triangulation, resolve_data_dir


@dataclass
class ClassicEDMDConfig:
    """Configuration for POD-reduced classical polynomial EDMD."""

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
    dictionary_degrees: tuple[int, ...] = (1, 2, 3, 4)
    selected_degree: int = 3
    rcond: float = 1e-8
    reduced_frequency_factor: float = float(np.pi / 2.0)
    dt: float = 0.07295943052
    zoom_xlim: tuple[float, float] = (-0.2, 1.2)
    zoom_ylim: tuple[float, float] = (-0.4, 0.4)


def relative_error(reference: np.ndarray, approximation: np.ndarray) -> float:
    """Return a relative Frobenius error."""

    denominator = np.linalg.norm(reference)
    if denominator == 0.0:
        return float(np.linalg.norm(reference - approximation))
    return float(np.linalg.norm(reference - approximation) / denominator)


def fit_models(
    normalized_states: np.ndarray,
    train_end: int,
    validation_end: int,
    degrees: tuple[int, ...],
    rcond: float,
) -> tuple[dict[int, object], list[dict[str, float | int]]]:
    """Evaluate dictionary degree on validation data, then refit through validation."""

    validation_rows: list[dict[str, float | int]] = []
    for degree in degrees:
        dictionary = PolynomialDictionary(
            n_states=normalized_states.shape[0],
            degree=degree,
            include_bias=True,
        )
        train_model = fit_edmd(
            normalized_states[:, : train_end - 1],
            normalized_states[:, 1:train_end],
            dictionary,
            rcond=rcond,
        )
        validation_reference = normalized_states[:, train_end:validation_end]
        validation_rollout = edmd_rollout(
            train_model,
            normalized_states[:, train_end],
            validation_end - train_end,
        )
        lifted_train = dictionary.transform(normalized_states[:, : train_end - 1])
        validation_rows.append(
            {
                "degree": degree,
                "feature_count": dictionary.n_features,
                "validation_rollout_relative_error": relative_error(
                    validation_reference,
                    validation_rollout,
                ),
                "lifted_training_condition_number": float(np.linalg.cond(lifted_train)),
            }
        )

    final_models: dict[int, object] = {}
    for degree in degrees:
        dictionary = PolynomialDictionary(
            n_states=normalized_states.shape[0],
            degree=degree,
            include_bias=True,
        )
        final_models[degree] = fit_edmd(
            normalized_states[:, : validation_end - 1],
            normalized_states[:, 1:validation_end],
            dictionary,
            rcond=rcond,
        )
    return final_models, validation_rows


def empirical_eigen_residuals(
    model: object,
    X: np.ndarray,
    Y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute empirical Koopman eigenfunction residuals on held-out pairs."""

    eigvals, left_vectors = np.linalg.eig(model.koopman_matrix.T)
    phi_x = model.dictionary.transform(X)
    phi_y = model.dictionary.transform(Y)
    residuals = np.empty(len(eigvals), dtype=float)
    for index, eigval in enumerate(eigvals):
        left = left_vectors[:, index].T
        value_x = left @ phi_x
        value_y = left @ phi_y
        denominator = np.linalg.norm(value_x)
        residuals[index] = float(
            np.linalg.norm(value_y - eigval * value_x)
            / max(denominator, np.finfo(float).eps)
        )
    return eigvals, residuals


def spectral_rows(
    eigvals: np.ndarray,
    residuals: np.ndarray,
    dt: float,
    reduced_frequency_factor: float,
    maximum_frequency: float = 2.0,
) -> list[dict[str, float | int]]:
    """Return positive-frequency, near-unit EDMD eigenpairs."""

    continuous = np.log(eigvals.astype(complex)) / dt
    reduced_frequency = continuous.imag / (2.0 * np.pi) * reduced_frequency_factor
    growth_rate = continuous.real
    rows: list[dict[str, float | int]] = []
    for index, eigval in enumerate(eigvals):
        frequency = float(reduced_frequency[index])
        if eigval.imag <= 1e-9 or abs(eigval) < 0.90 or frequency > maximum_frequency:
            continue
        rows.append(
            {
                "index_1based": index + 1,
                "reduced_frequency": frequency,
                "growth_rate": float(growth_rate[index]),
                "eigenvalue_radius": float(abs(eigval)),
                "empirical_residual": float(residuals[index]),
            }
        )
    rows.sort(key=lambda row: float(row["reduced_frequency"]))
    return rows


def unnormalize_states(
    normalized_states: np.ndarray,
    state_mean: np.ndarray,
    state_scale: np.ndarray,
) -> np.ndarray:
    """Convert normalized POD coordinates back to POD coefficients."""

    return normalized_states * state_scale + state_mean


def pressure_from_states(
    normalized_states: np.ndarray,
    state_mean: np.ndarray,
    state_scale: np.ndarray,
    pod_modes: np.ndarray,
) -> np.ndarray:
    """Map normalized POD coordinates to pressure fluctuations."""

    coefficients = unnormalize_states(normalized_states, state_mean, state_scale)
    return pod_modes @ coefficients


def save_degree_sensitivity(
    path: Path,
    validation_rows: list[dict[str, float | int]],
) -> None:
    """Plot validation accuracy and dictionary conditioning versus degree."""

    degrees = [int(row["degree"]) for row in validation_rows]
    errors = [float(row["validation_rollout_relative_error"]) for row in validation_rows]
    conditions = [float(row["lifted_training_condition_number"]) for row in validation_rows]

    fig, ax1 = plt.subplots(figsize=(7.0, 4.2), dpi=180)
    ax1.semilogy(degrees, errors, "o-", linewidth=1.2, label="validation rollout error")
    ax1.set_xlabel("Polynomial degree")
    ax1.set_ylabel("Validation relative error")
    ax1.set_xticks(degrees)
    ax1.grid(True, which="both", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.semilogy(degrees, conditions, "s--", color="tab:red", linewidth=1.1, label="condition number")
    ax2.set_ylabel("Condition number of lifted snapshots")
    ax1.set_title("EDMD dictionary-degree sensitivity")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="center right")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_eigenvalue_plot(
    path: Path,
    eigvals: np.ndarray,
    residuals: np.ndarray,
) -> None:
    """Plot EDMD eigenvalues colored by held-out empirical residual."""

    theta = np.linspace(0.0, 2.0 * np.pi, 500)
    color = np.log10(np.maximum(residuals, 1e-12))
    fig, ax = plt.subplots(figsize=(5.6, 5.1), dpi=180)
    ax.plot(np.cos(theta), np.sin(theta), "k--", linewidth=1.0, label="unit circle")
    scatter = ax.scatter(eigvals.real, eigvals.imag, c=color, cmap="viridis", s=34)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Re(lambda)")
    ax.set_ylabel("Im(lambda)")
    ax.set_title("Classical EDMD eigenvalues")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left")
    fig.colorbar(scatter, ax=ax, shrink=0.82, label="log10 empirical residual")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_harmonic_plot(
    path: Path,
    rows: list[dict[str, float | int]],
) -> None:
    """Plot radii of near-unit EDMD eigenvalues versus reduced frequency."""

    frequencies = [float(row["reduced_frequency"]) for row in rows]
    radii = [float(row["eigenvalue_radius"]) for row in rows]
    residuals = [float(row["empirical_residual"]) for row in rows]

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=180)
    scatter = ax.scatter(frequencies, radii, c=np.log10(np.maximum(residuals, 1e-12)), cmap="viridis", s=42)
    for harmonic in (0.186, 0.372, 0.558, 0.744):
        ax.axvline(harmonic, color="tab:red", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Reduced frequency")
    ax.set_ylabel("Eigenvalue radius")
    ax.set_title("Near-unit EDMD spectrum and buffet harmonics")
    ax.grid(True, alpha=0.25)
    fig.colorbar(scatter, ax=ax, shrink=0.82, label="log10 empirical residual")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_state_rollout_plot(
    path: Path,
    time: np.ndarray,
    reference: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> None:
    """Plot held-out POD-coordinate trajectories."""

    count = reference.shape[0]
    fig, axes = plt.subplots(count, 1, figsize=(9.0, 1.8 * count), dpi=180, sharex=True)
    colors = {"Degree 1": "tab:gray", "Degree 2": "tab:orange", "Degree 3": "tab:blue", "Degree 4": "tab:green"}
    for state_index, ax in enumerate(np.atleast_1d(axes)):
        ax.plot(time, reference[state_index], color="black", linewidth=1.1, label="CFD/POD state")
        for label, prediction in predictions.items():
            ax.plot(time, prediction[state_index], linewidth=0.85, color=colors.get(label), label=label)
        ax.set_ylabel(f"z{state_index + 1}")
        ax.grid(True, alpha=0.25)
    np.atleast_1d(axes)[0].legend(ncol=5, fontsize=8)
    np.atleast_1d(axes)[-1].set_xlabel("time")
    fig.suptitle("Held-out rollout of normalized POD states")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_phase_portrait(
    path: Path,
    reference: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> None:
    """Plot the first two POD coordinates as a phase portrait."""

    fig, ax = plt.subplots(figsize=(5.3, 4.8), dpi=180)
    ax.plot(reference[0], reference[1], color="black", linewidth=1.2, label="CFD/POD state")
    for label in ("Degree 1", "Degree 2", "Degree 3"):
        prediction = predictions[label]
        ax.plot(prediction[0], prediction[1], linewidth=0.9, label=label)
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    ax.set_title("EDMD phase portrait on held-out interval")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_error_time_plot(
    path: Path,
    time: np.ndarray,
    reference: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> None:
    """Plot state-space rollout error at each held-out time."""

    denominator = np.maximum(np.linalg.norm(reference, axis=0), np.finfo(float).eps)
    fig, ax = plt.subplots(figsize=(8.2, 4.0), dpi=180)
    for label, prediction in predictions.items():
        error = np.linalg.norm(reference - prediction, axis=0) / denominator
        ax.plot(time, error, linewidth=0.9, label=label)
    ax.set_xlabel("time")
    ax.set_ylabel("Relative state error")
    ax.set_title("EDMD held-out rollout error")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_metric_plot(
    path: Path,
    rows: list[dict[str, float | int]],
) -> None:
    """Plot test one-step, rollout, and pressure errors by dictionary degree."""

    labels = [f"degree {int(row['degree'])}" for row in rows]
    one_step = [float(row["test_one_step_relative_error"]) for row in rows]
    rollout = [float(row["test_rollout_relative_error"]) for row in rows]
    pressure = [float(row["cfd_pressure_fluctuation_relative_error"]) for row in rows]
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8.0, 4.3), dpi=180)
    ax.bar(x - width, one_step, width, label="one-step state")
    ax.bar(x, rollout, width, label="rollout state")
    ax.bar(x + width, pressure, width, label="CFD pressure fluctuation")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Relative error")
    ax.set_title("Classical EDMD error comparison")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_observation_histories(
    path: Path,
    time: np.ndarray,
    true_pressure: np.ndarray,
    pod_pressure: np.ndarray,
    degree1_pressure: np.ndarray,
    selected_pressure: np.ndarray,
    observation_indices: np.ndarray,
) -> None:
    """Compare pressure histories at the previously selected shock points."""

    labels = ("C", "D", "E")
    fig, axes = plt.subplots(3, 1, figsize=(8.8, 6.8), dpi=180, sharex=True)
    for ax, label, index in zip(axes, labels, observation_indices):
        ax.plot(time, true_pressure[index], color="black", linewidth=1.0, label="CFD")
        ax.plot(time, pod_pressure[index], "--", linewidth=0.85, label="POD rank 5")
        ax.plot(time, degree1_pressure[index], ":", linewidth=0.9, label="degree 1")
        ax.plot(time, selected_pressure[index], "-.", linewidth=0.9, label="degree 3 EDMD")
        ax.set_ylabel(f"p at {label}")
        ax.grid(True, alpha=0.25)
    axes[0].legend(ncol=4, fontsize=8)
    axes[-1].set_xlabel("time")
    fig.suptitle("Pressure histories on the held-out interval")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_pressure_snapshots(
    path: Path,
    triangulation: mtri.Triangulation,
    time: np.ndarray,
    true_fluctuation: np.ndarray,
    pod_fluctuation: np.ndarray,
    degree1_fluctuation: np.ndarray,
    selected_fluctuation: np.ndarray,
    indices: tuple[int, int],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    """Compare pressure-fluctuation fields at two held-out instants."""

    datasets = (
        ("CFD", true_fluctuation),
        ("POD rank 5", pod_fluctuation),
        ("degree 1", degree1_fluctuation),
        ("degree 3 EDMD", selected_fluctuation),
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
    fig.suptitle("Classical EDMD pressure-field reconstruction")
    fig.subplots_adjust(left=0.05, right=0.89, bottom=0.10, top=0.88, wspace=0.12, hspace=0.20)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write dictionary rows to a UTF-8 CSV file."""

    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_classic_edmd_analysis(
    config: ClassicEDMDConfig,
    figures_dir: Path,
    results_dir: Path,
) -> dict[str, object]:
    """Run classical polynomial EDMD and reconstruct OAT15A pressure fields."""

    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    reduced_data = np.load(config.pod_dmd_results)
    pod_coefficients = reduced_data["pod_coefficients"][: config.state_rank]
    pod_modes = reduced_data["pod_modes"][:, : config.state_rank]
    mean_pressure = reduced_data["mean_pressure"]
    snapshot_time = reduced_data["snapshot_time"]
    coordinates = reduced_data["coordinates"]
    nodes = reduced_data["nodes"]
    observation_indices = reduced_data["observation_indices"]

    snapshot_count = pod_coefficients.shape[1]
    train_end = int(snapshot_count * config.training_fraction)
    validation_end = int(snapshot_count * (config.training_fraction + config.validation_fraction))
    if not (2 <= train_end < validation_end < snapshot_count):
        raise ValueError("Invalid training/validation fractions")

    state_mean = pod_coefficients[:, :train_end].mean(axis=1, keepdims=True)
    state_scale = pod_coefficients[:, :train_end].std(axis=1, keepdims=True)
    state_scale = np.maximum(state_scale, np.finfo(float).eps)
    normalized_states = (pod_coefficients - state_mean) / state_scale

    models, validation_rows = fit_models(
        normalized_states,
        train_end,
        validation_end,
        config.dictionary_degrees,
        config.rcond,
    )
    if config.selected_degree not in models:
        raise ValueError("selected_degree must be included in dictionary_degrees")

    test_reference = normalized_states[:, validation_end:]
    test_x = normalized_states[:, validation_end:-1]
    test_y = normalized_states[:, validation_end + 1 :]
    rollout_steps = snapshot_count - validation_end
    predictions: dict[str, np.ndarray] = {}
    metric_rows: list[dict[str, float | int]] = []

    pod_config = PodConfig(data_root=config.data_root, data_dir=config.data_dir)
    data_dir = resolve_data_dir(pod_config)
    _, _, pressure_full, _ = load_oat15a_data(data_dir)
    pressure = pressure_full[:, :: config.stride]
    del pressure_full
    true_fluctuation = pressure[:, validation_end:] - mean_pressure[:, None]
    pod_test_fluctuation = pod_modes @ pod_coefficients[:, validation_end:]

    pressure_predictions: dict[int, np.ndarray] = {}
    for degree in config.dictionary_degrees:
        model = models[degree]
        one_step_prediction = edmd_one_step(model, test_x)
        rollout_prediction = edmd_rollout(
            model,
            normalized_states[:, validation_end],
            rollout_steps,
        )
        predictions[f"Degree {degree}"] = rollout_prediction
        predicted_fluctuation = pressure_from_states(
            rollout_prediction,
            state_mean,
            state_scale,
            pod_modes,
        )
        pressure_predictions[degree] = predicted_fluctuation
        metric_rows.append(
            {
                "degree": degree,
                "feature_count": model.dictionary.n_features,
                "test_one_step_relative_error": relative_error(test_y, one_step_prediction),
                "test_rollout_relative_error": relative_error(test_reference, rollout_prediction),
                "pod_subspace_pressure_relative_error": relative_error(
                    pod_test_fluctuation,
                    predicted_fluctuation,
                ),
                "cfd_pressure_fluctuation_relative_error": relative_error(
                    true_fluctuation,
                    predicted_fluctuation,
                ),
                "maximum_eigenvalue_radius": float(
                    np.max(np.abs(np.linalg.eigvals(model.koopman_matrix)))
                ),
            }
        )

    selected_model = models[config.selected_degree]
    eigvals, residuals = empirical_eigen_residuals(selected_model, test_x, test_y)
    spectrum = spectral_rows(
        eigvals,
        residuals,
        config.dt,
        config.reduced_frequency_factor,
    )

    triangulation = make_triangulation(coordinates, nodes)
    test_time = snapshot_time[validation_end:]
    print(f"POD state rank: {config.state_rank}", flush=True)
    print(f"Train/validation/test snapshots: {train_end}/{validation_end - train_end}/{snapshot_count - validation_end}", flush=True)
    print(f"Selected polynomial degree: {config.selected_degree}", flush=True)

    save_degree_sensitivity(figures_dir / "edmd_degree_sensitivity.png", validation_rows)
    save_eigenvalue_plot(figures_dir / "edmd_eigenvalues.png", eigvals, residuals)
    save_harmonic_plot(figures_dir / "edmd_harmonic_spectrum.png", spectrum)
    save_state_rollout_plot(
        figures_dir / "edmd_pod_state_rollout.png",
        test_time,
        test_reference,
        predictions,
    )
    save_phase_portrait(figures_dir / "edmd_phase_portrait.png", test_reference, predictions)
    save_error_time_plot(
        figures_dir / "edmd_rollout_error_time.png",
        test_time,
        test_reference,
        predictions,
    )
    save_metric_plot(figures_dir / "edmd_error_comparison.png", metric_rows)

    true_test_pressure = pressure[:, validation_end:]
    pod_test_pressure = mean_pressure[:, None] + pod_test_fluctuation
    degree1_pressure = mean_pressure[:, None] + pressure_predictions[1]
    selected_pressure = mean_pressure[:, None] + pressure_predictions[config.selected_degree]
    save_observation_histories(
        figures_dir / "edmd_observation_histories.png",
        test_time,
        true_test_pressure,
        pod_test_pressure,
        degree1_pressure,
        selected_pressure,
        observation_indices,
    )
    comparison_indices = (0, min(150, rollout_steps - 1))
    save_pressure_snapshots(
        figures_dir / "edmd_pressure_snapshot_comparison.png",
        triangulation,
        test_time,
        true_fluctuation,
        pod_test_fluctuation,
        pressure_predictions[1],
        pressure_predictions[config.selected_degree],
        comparison_indices,
        config.zoom_xlim,
        config.zoom_ylim,
    )

    np.savez_compressed(
        results_dir / "oat15a_classic_edmd_results.npz",
        snapshot_time=snapshot_time,
        state_mean=state_mean,
        state_scale=state_scale,
        normalized_states=normalized_states,
        test_reference=test_reference,
        degree1_prediction=predictions["Degree 1"],
        selected_prediction=predictions[f"Degree {config.selected_degree}"],
        selected_koopman_matrix=selected_model.koopman_matrix,
        selected_decoder=selected_model.decoder,
        selected_eigvals=eigvals,
        selected_residuals=residuals,
        pod_modes=pod_modes,
        mean_pressure=mean_pressure,
        observation_indices=observation_indices,
    )
    write_csv(results_dir / "degree_validation.csv", validation_rows)
    write_csv(results_dir / "degree_test_metrics.csv", metric_rows)
    write_csv(results_dir / "edmd_spectrum.csv", spectrum)

    pod_bound_error = relative_error(true_fluctuation, pod_test_fluctuation)
    selected_metrics = next(
        row for row in metric_rows if int(row["degree"]) == config.selected_degree
    )
    summary: dict[str, object] = {
        "method": "classical EDMD with an explicit polynomial dictionary",
        "state_definition": f"first {config.state_rank} POD pressure coefficients",
        "pod_energy_reference": "first 5 POD modes retain 99.37% fluctuation energy",
        "snapshot_count": snapshot_count,
        "train_snapshot_count": train_end,
        "validation_snapshot_count": validation_end - train_end,
        "test_snapshot_count": snapshot_count - validation_end,
        "state_normalization": "component-wise mean and standard deviation from training interval",
        "degree_validation": validation_rows,
        "degree_test_metrics": metric_rows,
        "selected_degree": config.selected_degree,
        "selected_feature_count": selected_model.dictionary.n_features,
        "selected_dictionary_terms": selected_model.dictionary.names(
            [f"z{i + 1}" for i in range(config.state_rank)]
        ),
        "selected_spectrum": spectrum,
        "pod_rank5_cfd_pressure_fluctuation_error": pod_bound_error,
        "selected_test_rollout_error": selected_metrics["test_rollout_relative_error"],
        "selected_cfd_pressure_fluctuation_error": selected_metrics[
            "cfd_pressure_fluctuation_relative_error"
        ],
        "figures_dir": str(figures_dir),
        "results_file": str(results_dir / "oat15a_classic_edmd_results.npz"),
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
