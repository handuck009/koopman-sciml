"""POD-DMD comparison for the OAT15A transonic buffet pressure snapshots.

The workflow follows Kou et al. (2016): POD is applied to pressure
fluctuations, while projected DMD is applied to time-shifted raw-pressure
snapshots so that the zero-frequency mode represents the mean flow.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

from oat15a_pod import (
    PodConfig,
    coefficient_fft_peaks,
    load_oat15a_data,
    make_triangulation,
    pressure_fluctuation,
    resolve_data_dir,
    save_coefficients_plot,
    save_energy_plot,
    save_fft_plot,
    save_field_plot,
    snapshot_pod,
    time_axis,
)


@dataclass
class ComparisonConfig:
    """Configuration for the stride-sampled POD-DMD comparison."""

    data_root: str = "F:/MY JOURNEY/data-driven_model"
    data_dir: str | None = None
    base_dt: float = 0.03647971526
    snapshot_start_time: float = 110.28
    stride: int = 2
    reduced_frequency_factor: float = float(np.pi / 2.0)
    pod_rank: int = 5
    dmd_fit_rank: int = 40
    dmd_pair_count: int = 4
    dmd_svd_max_rank: int = 40
    rank_sensitivity: tuple[int, ...] = (10, 20, 40)
    modes_to_plot: int = 5
    modes_to_save: int = 20
    zoom_xlim: tuple[float, float] = (-0.2, 1.2)
    zoom_ylim: tuple[float, float] = (-0.4, 0.4)


@dataclass
class SnapshotSVD:
    """Leading factors of an economy SVD computed by the snapshot method."""

    left: np.ndarray
    singular_values: np.ndarray
    right: np.ndarray


@dataclass
class ProjectedDMDResult:
    """Projected DMD result matching the mode definition in the paper."""

    eigvals: np.ndarray
    modes: np.ndarray
    amplitudes: np.ndarray
    a_tilde: np.ndarray
    rank: int


def truncated_snapshot_svd(X: np.ndarray, max_rank: int) -> SnapshotSVD:
    """Compute leading SVD factors from the temporal Gram matrix ``X.T @ X``."""

    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional snapshot matrix")
    if max_rank < 1:
        raise ValueError("max_rank must be positive")

    gram = X.T @ X
    eigvals, right = np.linalg.eigh(gram)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 0.0)
    right = right[:, order]
    singular_values = np.sqrt(eigvals)

    tolerance = np.finfo(float).eps * max(X.shape) * singular_values[0]
    numerical_rank = int(np.sum(singular_values > tolerance))
    rank = min(max_rank, numerical_rank)
    if rank < 1:
        raise ValueError("The snapshot matrix has zero numerical rank")

    singular_values = singular_values[:rank]
    right = right[:, :rank]
    left = (X @ right) / singular_values
    return SnapshotSVD(left=left, singular_values=singular_values, right=right)


def projected_dmd_from_svd(
    X: np.ndarray,
    Y: np.ndarray,
    svd: SnapshotSVD,
    rank: int,
) -> ProjectedDMDResult:
    """Fit projected DMD using ``Phi = U_r W`` as in Eq. (14) of the paper."""

    if X.shape != Y.shape:
        raise ValueError(f"X and Y must have equal shape, got {X.shape} and {Y.shape}")
    if rank < 1 or rank > len(svd.singular_values):
        raise ValueError(f"rank must be in [1, {len(svd.singular_values)}], got {rank}")

    U_r = svd.left[:, :rank]
    singular_values = svd.singular_values[:rank]
    V_r = svd.right[:, :rank]

    y_v_sigma_inv = (Y @ V_r) / singular_values
    a_tilde = U_r.T @ y_v_sigma_inv
    eigvals, eigenvectors = np.linalg.eig(a_tilde)
    modes = U_r @ eigenvectors

    # Since Phi = U W and U has orthonormal columns, W b = U^T x_1.
    modal_initial_state = U_r.T @ X[:, 0]
    amplitudes = np.linalg.lstsq(eigenvectors, modal_initial_state, rcond=None)[0]
    return ProjectedDMDResult(
        eigvals=eigvals,
        modes=modes,
        amplitudes=amplitudes,
        a_tilde=a_tilde,
        rank=rank,
    )


def dmd_spectral_quantities(
    eigvals: np.ndarray,
    dt: float,
    reduced_frequency_factor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return growth rate, signed cyclic frequency, and reduced frequency."""

    continuous = np.log(eigvals.astype(complex)) / dt
    growth_rate = continuous.real
    cyclic_frequency = continuous.imag / (2.0 * np.pi)
    reduced_frequency = cyclic_frequency * reduced_frequency_factor
    return growth_rate, cyclic_frequency, reduced_frequency


def conjugate_pairs(eigvals: np.ndarray) -> list[tuple[int, int]]:
    """Match positive-frequency eigenvalues with their complex conjugates."""

    positive = np.where(eigvals.imag > 1e-10)[0]
    negative = set(np.where(eigvals.imag < -1e-10)[0].tolist())
    pairs: list[tuple[int, int]] = []

    for positive_index in positive:
        if not negative:
            break
        negative_index = min(
            negative,
            key=lambda index: abs(eigvals[index] - np.conj(eigvals[positive_index])),
        )
        pairs.append((int(positive_index), int(negative_index)))
        negative.remove(negative_index)
    return pairs


def select_paper_modes(
    result: ProjectedDMDResult,
    pair_count: int,
    minimum_radius: float = 0.95,
) -> tuple[int, list[tuple[int, int]], np.ndarray]:
    """Select a static mode and dominant near-neutral conjugate pairs.

    The paper first identifies modes close to the unit circle and then ranks
    their contribution. Filtering strongly decaying modes prevents a transient
    numerical component from replacing a buffet harmonic.
    """

    static_candidates = np.where(np.abs(result.eigvals.imag) <= 1e-10)[0]
    if len(static_candidates):
        static_index = int(
            static_candidates[np.argmin(np.abs(result.eigvals[static_candidates] - 1.0))]
        )
    else:
        static_index = int(np.argmin(np.abs(result.eigvals - 1.0)))

    all_pairs = conjugate_pairs(result.eigvals)
    eligible_pairs = [
        pair
        for pair in all_pairs
        if float(np.mean(np.abs(result.eigvals[list(pair)]))) >= minimum_radius
    ]
    if len(eligible_pairs) < pair_count:
        eligible_pairs = all_pairs

    scored_pairs: list[tuple[float, tuple[int, int]]] = []
    for pair in eligible_pairs:
        score = float(np.linalg.norm(result.amplitudes[list(pair)]))
        scored_pairs.append((score, pair))
    scored_pairs.sort(key=lambda item: item[0], reverse=True)

    selected_pairs = [pair for _, pair in scored_pairs[:pair_count]]
    selected_pairs.sort(key=lambda pair: abs(np.angle(result.eigvals[pair[0]])))
    selected_indices = np.array(
        [static_index] + [index for pair in selected_pairs for index in pair],
        dtype=int,
    )
    return static_index, selected_pairs, selected_indices


def reconstruct_dmd(
    result: ProjectedDMDResult,
    steps: int,
    selected_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Reconstruct snapshots using all or selected DMD modes."""

    if selected_indices is None:
        selected_indices = np.arange(result.rank)
    eigvals = result.eigvals[selected_indices]
    amplitudes = result.amplitudes[selected_indices]
    modes = result.modes[:, selected_indices]
    powers = np.arange(steps)
    dynamics = amplitudes[:, None] * eigvals[:, None] ** powers[None, :]
    return (modes @ dynamics).real


def relative_frobenius_error(reference: np.ndarray, approximation: np.ndarray) -> float:
    """Return the relative Frobenius reconstruction error."""

    denominator = np.linalg.norm(reference)
    if denominator == 0.0:
        return float(np.linalg.norm(reference - approximation))
    return float(np.linalg.norm(reference - approximation) / denominator)


def nearest_node(coordinates: np.ndarray, target: tuple[float, float]) -> int:
    """Return the mesh-node index nearest to a target coordinate."""

    squared_distance = np.sum((coordinates - np.asarray(target)) ** 2, axis=1)
    return int(np.argmin(squared_distance))


def select_observation_points(
    coordinates: np.ndarray,
    fluctuation: np.ndarray,
) -> dict[str, int]:
    """Choose one smooth point and two points around the strongest shock motion."""

    standard_deviation = fluctuation.std(axis=1)
    search_mask = (
        (coordinates[:, 0] > 0.25)
        & (coordinates[:, 0] < 0.75)
        & (coordinates[:, 1] > 0.05)
        & (coordinates[:, 1] < 0.40)
    )
    candidates = np.where(search_mask)[0]
    shock_center = int(candidates[np.argmax(standard_deviation[candidates])])
    shock_x, shock_y = coordinates[shock_center]

    return {
        "C": nearest_node(coordinates, (float(shock_x), float(min(shock_y + 0.16, 0.38)))),
        "D": nearest_node(coordinates, (float(shock_x - 0.025), float(shock_y))),
        "E": nearest_node(coordinates, (float(shock_x + 0.025), float(shock_y))),
    }


def oriented_real_mode(mode: np.ndarray) -> np.ndarray:
    """Rotate a complex mode so its largest component is positive and real."""

    peak_index = int(np.argmax(np.abs(mode)))
    phase = np.exp(-1j * np.angle(mode[peak_index]))
    return (mode * phase).real


def save_dmd_eigenvalue_plot(
    path: Path,
    result: ProjectedDMDResult,
    selected_indices: np.ndarray,
) -> None:
    """Plot DMD eigenvalues and the unit circle."""

    theta = np.linspace(0.0, 2.0 * np.pi, 500)
    fig, ax = plt.subplots(figsize=(5.3, 5.0), dpi=180)
    ax.plot(np.cos(theta), np.sin(theta), "k--", linewidth=1.0, label="unit circle")
    ax.scatter(result.eigvals.real, result.eigvals.imag, s=30, facecolors="none", edgecolors="0.45")
    ax.scatter(
        result.eigvals[selected_indices].real,
        result.eigvals[selected_indices].imag,
        s=36,
        color="tab:red",
        label="selected modes",
    )
    ax.axhline(0.0, color="0.75", linewidth=0.7)
    ax.axvline(0.0, color="0.75", linewidth=0.7)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Re(lambda)")
    ax.set_ylabel("Im(lambda)")
    ax.set_title("DMD eigenvalues")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_dmd_frequency_plot(
    path: Path,
    rows: list[dict[str, float | int | str]],
) -> None:
    """Plot selected DMD modal amplitudes against reduced frequency."""

    pair_rows = [row for row in rows if row["kind"] == "conjugate_pair"]
    frequencies = [float(row["reduced_frequency"]) for row in pair_rows]
    amplitudes = [float(row["pair_amplitude"]) for row in pair_rows]

    fig, ax = plt.subplots(figsize=(6.8, 4.1), dpi=180)
    markerline, stemlines, baseline = ax.stem(frequencies, amplitudes)
    plt.setp(markerline, markersize=6, color="tab:red")
    plt.setp(stemlines, linewidth=1.4, color="tab:red")
    plt.setp(baseline, linewidth=0.8, color="0.4")
    for frequency, amplitude in zip(frequencies, amplitudes):
        ax.annotate(f"k={frequency:.3f}", (frequency, amplitude), xytext=(4, 5), textcoords="offset points")
    ax.set_xlabel("Reduced frequency")
    ax.set_ylabel("Pair amplitude")
    ax.set_title("Selected DMD frequency pairs")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_dmd_coefficients_plot(
    path: Path,
    time: np.ndarray,
    result: ProjectedDMDResult,
    pairs: list[tuple[int, int]],
) -> None:
    """Plot temporal coefficients of positive-frequency DMD representatives."""

    fig, axes = plt.subplots(len(pairs), 1, figsize=(8.4, 1.7 * len(pairs)), dpi=180, sharex=True)
    axes = np.atleast_1d(axes)
    powers = np.arange(len(time))
    for pair_index, (mode_index, _) in enumerate(pairs):
        coefficient = result.amplitudes[mode_index] * result.eigvals[mode_index] ** powers
        axes[pair_index].plot(time, coefficient.real, linewidth=0.9)
        axes[pair_index].set_ylabel(f"pair {pair_index + 1}")
        axes[pair_index].grid(True, alpha=0.25)
    axes[-1].set_xlabel("time")
    fig.suptitle("Selected DMD modal coefficients")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_observation_map(
    path: Path,
    triangulation: mtri.Triangulation,
    coordinates: np.ndarray,
    mean_pressure: np.ndarray,
    observation_points: dict[str, int],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    """Plot automatically selected observation points around the shock region."""

    fig, ax = plt.subplots(figsize=(8.0, 3.4), dpi=180)
    field = ax.tripcolor(triangulation, mean_pressure, shading="gouraud", cmap="viridis")
    offsets = {"C": (8, 8), "D": (-18, 8), "E": (8, 8)}
    for label, index in observation_points.items():
        x, y = coordinates[index]
        ax.scatter(x, y, s=38, edgecolor="white", linewidth=0.8, label=label)
        ax.annotate(
            label,
            (x, y),
            xytext=offsets.get(label, (5, 5)),
            textcoords="offset points",
            color="white",
            weight="bold",
        )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Observation points near the moving shock")
    fig.colorbar(field, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_observation_histories(
    path: Path,
    time: np.ndarray,
    pressure: np.ndarray,
    pod_reconstruction: np.ndarray,
    dmd_reconstruction: np.ndarray,
    observation_points: dict[str, int],
) -> dict[str, dict[str, float]]:
    """Compare true, POD, and DMD pressure histories at three points."""

    fig, axes = plt.subplots(3, 1, figsize=(8.5, 6.5), dpi=180, sharex=True)
    metrics: dict[str, dict[str, float]] = {}
    for ax, (label, index) in zip(axes, observation_points.items()):
        reference = pressure[index]
        pod_signal = pod_reconstruction[index]
        dmd_signal = dmd_reconstruction[index]
        ax.plot(time, reference, color="black", linewidth=1.0, label="CFD")
        ax.plot(time, pod_signal, "--", color="tab:blue", linewidth=0.9, label="POD")
        ax.plot(time, dmd_signal, "-.", color="tab:red", linewidth=0.9, label="DMD")
        ax.set_ylabel(f"p at {label}")
        ax.grid(True, alpha=0.25)
        metrics[label] = {
            "pod_rmse": float(np.sqrt(np.mean((reference - pod_signal) ** 2))),
            "dmd_rmse": float(np.sqrt(np.mean((reference - dmd_signal) ** 2))),
        }
    axes[0].legend(ncol=3)
    axes[-1].set_xlabel("time")
    fig.suptitle("Pressure histories at observation points")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return metrics


def save_time_error_plot(
    path: Path,
    time: np.ndarray,
    fluctuation: np.ndarray,
    pod_reconstruction: np.ndarray,
    dmd_reconstruction: np.ndarray,
    mean_pressure: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Plot snapshot-wise relative errors normalized by fluctuation magnitude."""

    pod_fluctuation = pod_reconstruction - mean_pressure[:, None]
    dmd_fluctuation = dmd_reconstruction - mean_pressure[:, None]
    denominator = np.linalg.norm(fluctuation, axis=0)
    denominator = np.maximum(denominator, np.finfo(float).eps)
    pod_error = np.linalg.norm(fluctuation - pod_fluctuation, axis=0) / denominator
    dmd_error = np.linalg.norm(fluctuation - dmd_fluctuation, axis=0) / denominator

    fig, ax = plt.subplots(figsize=(8.0, 4.0), dpi=180)
    ax.plot(time, pod_error, linewidth=0.9, label="POD rank 5")
    ax.plot(time, dmd_error, linewidth=0.9, label="DMD 9 selected modes")
    ax.set_xlabel("time")
    ax.set_ylabel("Relative fluctuation error")
    ax.set_title("Snapshot-wise reconstruction error")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return pod_error, dmd_error


def save_snapshot_comparison(
    path: Path,
    triangulation: mtri.Triangulation,
    time: np.ndarray,
    mean_pressure: np.ndarray,
    pressure: np.ndarray,
    pod_reconstruction: np.ndarray,
    dmd_reconstruction: np.ndarray,
    time_indices: tuple[int, int],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    """Compare true, POD, and DMD pressure fluctuations at two instants."""

    fig, axes = plt.subplots(2, 3, figsize=(12.0, 5.7), dpi=180, sharex=True, sharey=True)
    names = ("CFD", "POD", "DMD")
    datasets = (pressure, pod_reconstruction, dmd_reconstruction)
    selected_true_fields = [pressure[:, index] - mean_pressure for index in time_indices]
    vmax = max(float(np.nanpercentile(np.abs(field), 99.0)) for field in selected_true_fields)
    last_plot = None
    for row, time_index in enumerate(time_indices):
        for column, (name, data) in enumerate(zip(names, datasets)):
            values = data[:, time_index] - mean_pressure
            last_plot = axes[row, column].tripcolor(
                triangulation,
                values,
                shading="gouraud",
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
            )
            axes[row, column].set_xlim(*xlim)
            axes[row, column].set_ylim(*ylim)
            axes[row, column].set_aspect("equal", adjustable="box")
            axes[row, column].set_title(f"{name}, t={time[time_index]:.3f}")
    for ax in axes[:, 0]:
        ax.set_ylabel("y")
    for ax in axes[-1, :]:
        ax.set_xlabel("x")
    if last_plot is not None:
        colorbar_axis = fig.add_axes([0.90, 0.18, 0.018, 0.62])
        fig.colorbar(last_plot, cax=colorbar_axis, label="pressure fluctuation")
    fig.suptitle("Pressure-fluctuation reconstruction at two instants")
    fig.subplots_adjust(left=0.06, right=0.88, bottom=0.10, top=0.88, wspace=0.12, hspace=0.20)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_dmd_mode_count_error(
    path: Path,
    pressure: np.ndarray,
    result: ProjectedDMDResult,
    static_index: int,
    pairs: list[tuple[int, int]],
) -> list[dict[str, float | int]]:
    """Plot reconstruction loss as static and harmonic pairs are added."""

    selected = [static_index]
    rows: list[dict[str, float | int]] = []
    for pair_number in range(len(pairs) + 1):
        if pair_number > 0:
            selected.extend(pairs[pair_number - 1])
        reconstruction = reconstruct_dmd(result, pressure.shape[1], np.asarray(selected, dtype=int))
        error = relative_frobenius_error(pressure, reconstruction)
        rows.append(
            {
                "mode_count": len(selected),
                "conjugate_pair_count": pair_number,
                "raw_pressure_relative_error": error,
            }
        )

    fig, ax = plt.subplots(figsize=(6.6, 4.0), dpi=180)
    ax.plot(
        [int(row["mode_count"]) for row in rows],
        [100.0 * float(row["raw_pressure_relative_error"]) for row in rows],
        "o-",
        linewidth=1.2,
    )
    ax.set_xlabel("Selected DMD mode count")
    ax.set_ylabel("Raw-pressure reconstruction error (%)")
    ax.set_title("DMD mode count and reconstruction loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return rows


def save_error_summary(
    path: Path,
    metrics: dict[str, float],
) -> None:
    """Plot global POD-DMD reconstruction metrics."""

    labels = ["Raw pressure", "Pressure fluctuation"]
    pod_values = [metrics["pod_raw_relative_error"], metrics["pod_fluctuation_relative_error"]]
    dmd_values = [metrics["dmd_raw_relative_error"], metrics["dmd_fluctuation_relative_error"]]
    x = np.arange(len(labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(6.8, 4.2), dpi=180)
    ax.bar(x - width / 2, pod_values, width, label="POD rank 5")
    ax.bar(x + width / 2, dmd_values, width, label="DMD 9 modes")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Relative Frobenius error")
    ax.set_title("Global reconstruction comparison")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def selected_dmd_rows(
    result: ProjectedDMDResult,
    static_index: int,
    pairs: list[tuple[int, int]],
    dt: float,
    reduced_frequency_factor: float,
) -> list[dict[str, float | int | str]]:
    """Build a serializable table for the selected static and conjugate modes."""

    growth, _, reduced = dmd_spectral_quantities(result.eigvals, dt, reduced_frequency_factor)
    rows: list[dict[str, float | int | str]] = [
        {
            "kind": "static",
            "indices_1based": str(static_index + 1),
            "reduced_frequency": 0.0,
            "growth_rate": float(growth[static_index]),
            "eigenvalue_radius": float(abs(result.eigvals[static_index])),
            "pair_amplitude": float(abs(result.amplitudes[static_index])),
        }
    ]
    for pair in pairs:
        representative = pair[0]
        rows.append(
            {
                "kind": "conjugate_pair",
                "indices_1based": f"{pair[0] + 1}-{pair[1] + 1}",
                "reduced_frequency": float(abs(reduced[representative])),
                "growth_rate": float(np.mean(growth[list(pair)])),
                "eigenvalue_radius": float(np.mean(np.abs(result.eigvals[list(pair)]))),
                "pair_amplitude": float(np.linalg.norm(result.amplitudes[list(pair)])),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a list of dictionaries to CSV."""

    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_stride_pod_dmd_analysis(
    config: ComparisonConfig,
    figures_dir: Path,
    results_dir: Path,
) -> dict[str, object]:
    """Run stride-sampled POD, projected DMD, and their comparison."""

    if config.stride < 1:
        raise ValueError("stride must be at least 1")
    if config.dmd_svd_max_rank < max(config.rank_sensitivity + (config.dmd_fit_rank,)):
        raise ValueError("dmd_svd_max_rank must cover the fit and sensitivity ranks")

    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    pod_config = PodConfig(data_root=config.data_root, data_dir=config.data_dir)
    data_dir = resolve_data_dir(pod_config)
    coordinates, nodes, pressure_full, output = load_oat15a_data(data_dir)
    pressure = pressure_full[:, :: config.stride]
    del pressure_full

    dt = config.base_dt * config.stride
    snapshot_time = time_axis(pressure.shape[1], dt, config.snapshot_start_time)
    mean_pressure, fluctuation = pressure_fluctuation(pressure)
    triangulation = make_triangulation(coordinates, nodes)

    print(f"Data directory: {data_dir}", flush=True)
    print(f"Stride: {config.stride}", flush=True)
    print(f"Subsampled pressure shape: {pressure.shape}", flush=True)
    print(f"Effective dt: {dt:.11f}", flush=True)

    print("Computing snapshot POD on pressure fluctuations...", flush=True)
    pod_modes, pod_singular_values, pod_coefficients, pod_energy = snapshot_pod(fluctuation)
    pod_rank = min(config.pod_rank, pod_modes.shape[1])
    pod_reconstruction = mean_pressure[:, None] + pod_modes[:, :pod_rank] @ pod_coefficients[:pod_rank]

    print("Computing truncated snapshot SVD for projected DMD...", flush=True)
    X = pressure[:, :-1]
    Y = pressure[:, 1:]
    dmd_svd = truncated_snapshot_svd(X, config.dmd_svd_max_rank)
    dmd_result = projected_dmd_from_svd(X, Y, dmd_svd, config.dmd_fit_rank)
    static_index, selected_pairs, selected_indices = select_paper_modes(
        dmd_result,
        config.dmd_pair_count,
    )
    dmd_reconstruction = reconstruct_dmd(dmd_result, pressure.shape[1], selected_indices)
    dmd_rows = selected_dmd_rows(
        dmd_result,
        static_index,
        selected_pairs,
        dt,
        config.reduced_frequency_factor,
    )

    rank_sensitivity: list[dict[str, object]] = []
    for rank in config.rank_sensitivity:
        rank_result = projected_dmd_from_svd(X, Y, dmd_svd, rank)
        rank_static, rank_pairs, _ = select_paper_modes(rank_result, config.dmd_pair_count)
        rank_rows = selected_dmd_rows(
            rank_result,
            rank_static,
            rank_pairs,
            dt,
            config.reduced_frequency_factor,
        )
        rank_sensitivity.append(
            {
                "rank": rank,
                "selected_pair_frequencies": [
                    float(row["reduced_frequency"])
                    for row in rank_rows
                    if row["kind"] == "conjugate_pair"
                ],
            }
        )

    pod_fft_peaks = coefficient_fft_peaks(
        pod_coefficients,
        dt,
        config.reduced_frequency_factor,
        count=config.modes_to_plot,
    )
    observation_points = select_observation_points(coordinates, fluctuation)

    metrics = {
        "pod_raw_relative_error": relative_frobenius_error(pressure, pod_reconstruction),
        "dmd_raw_relative_error": relative_frobenius_error(pressure, dmd_reconstruction),
        "pod_fluctuation_relative_error": relative_frobenius_error(
            fluctuation,
            pod_reconstruction - mean_pressure[:, None],
        ),
        "dmd_fluctuation_relative_error": relative_frobenius_error(
            fluctuation,
            dmd_reconstruction - mean_pressure[:, None],
        ),
    }

    print("Saving POD and DMD figures...", flush=True)
    save_energy_plot(figures_dir / "pod_energy_spectrum.png", pod_energy, spectrum_modes=60)
    save_coefficients_plot(
        figures_dir / "pod_temporal_coefficients.png",
        snapshot_time,
        pod_coefficients,
        count=config.modes_to_plot,
    )
    save_fft_plot(
        figures_dir / "pod_temporal_coefficients_fft.png",
        pod_coefficients,
        dt,
        config.reduced_frequency_factor,
        count=config.modes_to_plot,
    )
    for mode_index in range(min(config.modes_to_plot, pod_modes.shape[1])):
        save_field_plot(
            figures_dir / f"pod_mode_{mode_index + 1:02d}_zoom.png",
            triangulation,
            pod_modes[:, mode_index],
            f"POD mode {mode_index + 1}, zoomed",
            xlim=config.zoom_xlim,
            ylim=config.zoom_ylim,
        )

    save_dmd_eigenvalue_plot(figures_dir / "dmd_eigenvalues.png", dmd_result, selected_indices)
    save_dmd_frequency_plot(figures_dir / "dmd_selected_frequencies.png", dmd_rows)
    save_dmd_coefficients_plot(
        figures_dir / "dmd_temporal_coefficients.png",
        snapshot_time,
        dmd_result,
        selected_pairs,
    )
    dmd_mode_count_errors = save_dmd_mode_count_error(
        figures_dir / "dmd_mode_count_error.png",
        pressure,
        dmd_result,
        static_index,
        selected_pairs,
    )
    save_field_plot(
        figures_dir / "dmd_static_mode_zoom.png",
        triangulation,
        oriented_real_mode(dmd_result.modes[:, static_index]),
        "DMD static mode, zoomed",
        cmap="viridis",
        symmetric=False,
        xlim=config.zoom_xlim,
        ylim=config.zoom_ylim,
    )
    for pair_number, pair in enumerate(selected_pairs, start=1):
        representative = pair[0]
        frequency = next(
            float(row["reduced_frequency"])
            for row in dmd_rows
            if row["indices_1based"] == f"{pair[0] + 1}-{pair[1] + 1}"
        )
        save_field_plot(
            figures_dir / f"dmd_pair_{pair_number:02d}_zoom.png",
            triangulation,
            oriented_real_mode(dmd_result.modes[:, representative]),
            f"DMD pair {pair_number}, k={frequency:.3f}",
            xlim=config.zoom_xlim,
            ylim=config.zoom_ylim,
        )

    save_observation_map(
        figures_dir / "observation_points.png",
        triangulation,
        coordinates,
        mean_pressure,
        observation_points,
        config.zoom_xlim,
        config.zoom_ylim,
    )
    observation_metrics = save_observation_histories(
        figures_dir / "pod_dmd_observation_histories.png",
        snapshot_time,
        pressure,
        pod_reconstruction,
        dmd_reconstruction,
        observation_points,
    )
    pod_time_error, dmd_time_error = save_time_error_plot(
        figures_dir / "pod_dmd_time_errors.png",
        snapshot_time,
        fluctuation,
        pod_reconstruction,
        dmd_reconstruction,
        mean_pressure,
    )
    save_error_summary(figures_dir / "pod_dmd_global_error_comparison.png", metrics)

    pod_node_rmse = np.sqrt(np.mean((pressure - pod_reconstruction) ** 2, axis=1))
    dmd_node_rmse = np.sqrt(np.mean((pressure - dmd_reconstruction) ** 2, axis=1))
    save_field_plot(
        figures_dir / "pod_node_rmse_zoom.png",
        triangulation,
        pod_node_rmse,
        "POD node-wise RMSE, zoomed",
        cmap="magma",
        symmetric=False,
        xlim=config.zoom_xlim,
        ylim=config.zoom_ylim,
    )
    save_field_plot(
        figures_dir / "dmd_node_rmse_zoom.png",
        triangulation,
        dmd_node_rmse,
        "DMD node-wise RMSE, zoomed",
        cmap="magma",
        symmetric=False,
        xlim=config.zoom_xlim,
        ylim=config.zoom_ylim,
    )

    target_times = (147.4075, 151.3555)
    comparison_indices = tuple(int(np.argmin(np.abs(snapshot_time - target))) for target in target_times)
    save_snapshot_comparison(
        figures_dir / "pod_dmd_snapshot_comparison.png",
        triangulation,
        snapshot_time,
        mean_pressure,
        pressure,
        pod_reconstruction,
        dmd_reconstruction,
        comparison_indices,
        config.zoom_xlim,
        config.zoom_ylim,
    )

    modes_to_save = min(config.modes_to_save, pod_modes.shape[1])
    np.savez_compressed(
        results_dir / "oat15a_stride2_pod_dmd_results.npz",
        snapshot_time=snapshot_time,
        mean_pressure=mean_pressure,
        pod_energy=pod_energy,
        pod_singular_values=pod_singular_values,
        pod_modes=pod_modes[:, :modes_to_save],
        pod_coefficients=pod_coefficients[:modes_to_save],
        dmd_eigvals=dmd_result.eigvals,
        dmd_modes=dmd_result.modes,
        dmd_amplitudes=dmd_result.amplitudes,
        dmd_selected_indices=selected_indices,
        pod_time_relative_error=pod_time_error,
        dmd_time_relative_error=dmd_time_error,
        pod_node_rmse=pod_node_rmse,
        dmd_node_rmse=dmd_node_rmse,
        observation_indices=np.array(list(observation_points.values()), dtype=int),
        coordinates=coordinates,
        nodes=nodes,
    )

    write_csv(results_dir / "dmd_selected_modes.csv", dmd_rows)
    write_csv(results_dir / "dmd_mode_count_errors.csv", dmd_mode_count_errors)
    write_csv(
        results_dir / "comparison_metrics.csv",
        [{"metric": key, "value": value} for key, value in metrics.items()],
    )

    cumulative_energy = np.cumsum(pod_energy)
    observation_coordinates = {
        label: {
            "node_index_0based": int(index),
            "coordinate": [float(value) for value in coordinates[index]],
        }
        for label, index in observation_points.items()
    }
    summary: dict[str, object] = {
        "reference_method": "Kou et al. 2016 POD-DMD transonic buffet comparison",
        "data_dir": str(data_dir),
        "original_snapshot_count": 2000,
        "stride": config.stride,
        "subsampled_pressure_shape": [int(value) for value in pressure.shape],
        "effective_dt": float(dt),
        "snapshot_start_time": float(snapshot_time[0]),
        "snapshot_end_time": float(snapshot_time[-1]),
        "pod_input": "mean-subtracted pressure fluctuation",
        "pod_rank": pod_rank,
        "pod_cumulative_energy": float(cumulative_energy[pod_rank - 1]),
        "pod_first_10_energy_fraction": [float(value) for value in pod_energy[:10]],
        "pod_fft_peaks": pod_fft_peaks,
        "dmd_input": "raw pressure, X=P[:, :-1], Y=P[:, 1:]",
        "dmd_mode_definition": "projected DMD, Phi = U_r W (paper Eq. 14)",
        "dmd_fit_rank": config.dmd_fit_rank,
        "dmd_selected_mode_count": int(len(selected_indices)),
        "dmd_selected_modes": dmd_rows,
        "dmd_mode_count_errors": dmd_mode_count_errors,
        "dmd_rank_sensitivity": rank_sensitivity,
        "global_reconstruction_metrics": metrics,
        "observation_points": observation_coordinates,
        "observation_point_rmse": observation_metrics,
        "comparison_snapshot_times": [float(snapshot_time[index]) for index in comparison_indices],
        "figures_dir": str(figures_dir),
        "results_file": str(results_dir / "oat15a_stride2_pod_dmd_results.npz"),
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
