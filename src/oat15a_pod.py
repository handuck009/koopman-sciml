"""POD analysis utilities for the OAT15A transonic buffet snapshots.

The pressure snapshot matrix has shape ``n_nodes x n_snapshots``. Each column is
one pressure field on the same unstructured quadrilateral mesh.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np


@dataclass
class PodConfig:
    """Configuration for full-snapshot POD analysis."""

    data_root: str = "F:/MY JOURNEY/data-driven_model"
    data_dir: str | None = None
    dt: float = 0.03647971526
    snapshot_start_time: float = 110.28
    reduced_frequency_factor: float = float(np.pi / 2.0)
    modes_to_plot: int = 8
    modes_to_save: int = 20
    spectrum_modes: int = 80
    fft_coefficients: int = 4
    energy_ranks: tuple[int, ...] = (1, 2, 4, 6, 8, 10, 20, 50, 100)
    zoom_xlim: tuple[float, float] = (-0.2, 1.2)
    zoom_ylim: tuple[float, float] = (-0.4, 0.4)


def find_oat15a_data_dir(root: Path) -> Path:
    """Find the OAT15A data directory without hard-coding Chinese path names."""

    assignment_dirs = [p for p in root.iterdir() if p.is_dir() and "DMD" in p.name]
    if not assignment_dirs:
        raise FileNotFoundError(f"Could not find an assignment directory containing 'DMD' under {root}")

    for assignment in assignment_dirs:
        data_dirs = [p for p in assignment.iterdir() if p.is_dir() and "OAT15A" in p.name]
        if data_dirs:
            return data_dirs[0]

    raise FileNotFoundError(f"Could not find an OAT15A data directory under {root}")


def resolve_data_dir(config: PodConfig) -> Path:
    """Resolve the configured data directory."""

    if config.data_dir:
        data_dir = Path(config.data_dir)
    else:
        data_dir = find_oat15a_data_dir(Path(config.data_root))

    required = ["coordinate.dat", "node.dat", "Snapshotp.dat", "OUTPUT.dat"]
    missing = [name for name in required if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing files in {data_dir}: {missing}")
    return data_dir


def load_oat15a_data(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load mesh, pressure snapshots, and lift coefficient output."""

    coordinates = np.loadtxt(data_dir / "coordinate.dat")
    nodes = np.loadtxt(data_dir / "node.dat", dtype=int) - 1
    pressure = np.loadtxt(data_dir / "Snapshotp.dat")
    output = np.loadtxt(data_dir / "OUTPUT.dat")

    if pressure.shape[0] != coordinates.shape[0]:
        raise ValueError(
            f"Pressure rows should match coordinate rows, got {pressure.shape[0]} and {coordinates.shape[0]}"
        )
    if nodes.min() < 0 or nodes.max() >= coordinates.shape[0]:
        raise ValueError("Connectivity indices are outside coordinate range.")
    return coordinates, nodes, pressure, output


def quads_to_triangles(nodes: np.ndarray) -> np.ndarray:
    """Split quadrilateral cells into triangles for Matplotlib plotting."""

    tri_1 = nodes[:, [0, 1, 2]]
    tri_2 = nodes[:, [0, 2, 3]]
    return np.vstack([tri_1, tri_2])


def make_triangulation(coordinates: np.ndarray, nodes: np.ndarray) -> mtri.Triangulation:
    """Build a triangular plotting mesh from the quadrilateral connectivity."""

    triangles = quads_to_triangles(nodes)
    return mtri.Triangulation(coordinates[:, 0], coordinates[:, 1], triangles)


def pressure_fluctuation(pressure: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return temporal mean pressure and pressure fluctuation snapshots."""

    mean_pressure = pressure.mean(axis=1, keepdims=True)
    fluctuation = pressure - mean_pressure
    return mean_pressure[:, 0], fluctuation


def snapshot_pod(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute exact POD by the snapshot method.

    Parameters
    ----------
    X:
        Mean-subtracted snapshot matrix with shape ``n_space x n_time``.

    Returns
    -------
    modes:
        Spatial POD modes, columns are modes.
    singular_values:
        Singular values of the snapshot matrix.
    coefficients:
        Temporal coefficients, rows are modes and columns are time snapshots.
    energy_fraction:
        Modal energy fractions.
    """

    n_time = X.shape[1]
    covariance = (X.T @ X) / (n_time - 1)
    eigvals, eigvecs = np.linalg.eigh(covariance)

    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 0.0)
    eigvecs = eigvecs[:, order]

    singular_values = np.sqrt(eigvals * (n_time - 1))
    tolerance = np.finfo(float).eps * max(X.shape) * singular_values[0]
    valid = singular_values > tolerance

    modes = np.zeros((X.shape[0], len(singular_values)), dtype=float)
    modes[:, valid] = (X @ eigvecs[:, valid]) / singular_values[valid]
    modes = orient_modes(modes)

    coefficients = modes.T @ X
    energy_fraction = eigvals / eigvals.sum()
    return modes, singular_values, coefficients, energy_fraction


def orient_modes(modes: np.ndarray) -> np.ndarray:
    """Make mode signs deterministic for stable plots."""

    oriented = modes.copy()
    for i in range(oriented.shape[1]):
        peak_index = np.argmax(np.abs(oriented[:, i]))
        if oriented[peak_index, i] < 0:
            oriented[:, i] *= -1.0
    return oriented


def time_axis(n_snapshots: int, dt: float, start_time: float) -> np.ndarray:
    """Build the snapshot time axis."""

    return start_time + np.arange(n_snapshots) * dt


def coefficient_fft_peaks(
    coefficients: np.ndarray,
    dt: float,
    reduced_frequency_factor: float,
    count: int,
) -> list[dict[str, float]]:
    """Find dominant nonzero FFT peaks of leading POD coefficients."""

    peaks: list[dict[str, float]] = []
    n_time = coefficients.shape[1]
    raw_freq = np.fft.rfftfreq(n_time, d=dt)

    for mode_index in range(min(count, coefficients.shape[0])):
        signal = coefficients[mode_index] - coefficients[mode_index].mean()
        amplitude = np.abs(np.fft.rfft(signal))
        if len(amplitude) <= 1:
            continue
        peak_index = int(np.argmax(amplitude[1:]) + 1)
        peaks.append(
            {
                "mode_index_1based": float(mode_index + 1),
                "raw_frequency": float(raw_freq[peak_index]),
                "reduced_frequency": float(raw_freq[peak_index] * reduced_frequency_factor),
                "fft_amplitude": float(amplitude[peak_index]),
            }
        )
    return peaks


def save_field_plot(
    path: Path,
    triangulation: mtri.Triangulation,
    field: np.ndarray,
    title: str,
    cmap: str = "RdBu_r",
    symmetric: bool = True,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Save one scalar field on the airfoil mesh."""

    fig, ax = plt.subplots(figsize=(8, 3.2), dpi=180)
    values = np.asarray(field).real
    if symmetric:
        vmax = np.nanpercentile(np.abs(values), 99.0)
        if not np.isfinite(vmax) or vmax == 0.0:
            vmax = np.nanmax(np.abs(values)) + 1e-12
        vmin = -vmax
    else:
        vmin, vmax = np.nanpercentile(values, [1.0, 99.0])
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin = np.nanmin(values)
            vmax = np.nanmax(values) + 1e-12

    field_plot = ax.tripcolor(
        triangulation,
        values,
        shading="gouraud",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidth=0.0,
    )
    ax.set_aspect("equal", adjustable="box")
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    fig.colorbar(field_plot, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_energy_plot(path: Path, energy: np.ndarray, spectrum_modes: int) -> None:
    """Save modal and cumulative POD energy spectrum."""

    n_modes = min(spectrum_modes, len(energy))
    modes = np.arange(1, n_modes + 1)

    fig, ax1 = plt.subplots(figsize=(7.2, 4.2), dpi=180)
    ax1.semilogy(modes, energy[:n_modes], "o-", linewidth=1.2, markersize=3.5, label="energy fraction")
    ax1.set_xlabel("POD mode index")
    ax1.set_ylabel("Energy fraction")
    ax1.grid(True, which="both", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(modes, np.cumsum(energy[:n_modes]), "s-", color="tab:orange", linewidth=1.0, markersize=3.0)
    ax2.set_ylabel("Cumulative energy")
    ax2.set_ylim(0.0, 1.02)

    ax1.set_title("POD energy spectrum")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_coefficients_plot(path: Path, time: np.ndarray, coefficients: np.ndarray, count: int) -> None:
    """Save leading POD temporal coefficients."""

    count = min(count, coefficients.shape[0])
    fig, axes = plt.subplots(count, 1, figsize=(8.4, 1.65 * count), dpi=180, sharex=True)
    for i, ax in enumerate(np.atleast_1d(axes)):
        ax.plot(time, coefficients[i], linewidth=0.9)
        ax.set_ylabel(f"a{i + 1}")
        ax.grid(True, alpha=0.25)
    np.atleast_1d(axes)[-1].set_xlabel("time")
    fig.suptitle("Leading POD temporal coefficients")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_fft_plot(
    path: Path,
    coefficients: np.ndarray,
    dt: float,
    reduced_frequency_factor: float,
    count: int,
) -> None:
    """Save FFT spectra of leading POD temporal coefficients."""

    count = min(count, coefficients.shape[0])
    raw_freq = np.fft.rfftfreq(coefficients.shape[1], d=dt)
    reduced_freq = raw_freq * reduced_frequency_factor

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=180)
    for i in range(count):
        signal = coefficients[i] - coefficients[i].mean()
        amplitude = np.abs(np.fft.rfft(signal))
        amplitude = amplitude / (amplitude.max() + 1e-15)
        ax.plot(reduced_freq[1:], amplitude[1:], linewidth=1.0, label=f"a{i + 1}")

    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Reduced frequency")
    ax.set_ylabel("Normalized FFT amplitude")
    ax.set_title("FFT of leading POD temporal coefficients")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_reconstruction_error_plot(path: Path, energy: np.ndarray, ranks: tuple[int, ...]) -> dict[str, float]:
    """Save low-rank relative reconstruction error estimated from POD energy."""

    valid_ranks = [rank for rank in ranks if rank <= len(energy)]
    cumulative = np.cumsum(energy)
    errors = {str(rank): float(np.sqrt(max(0.0, 1.0 - cumulative[rank - 1]))) for rank in valid_ranks}

    fig, ax = plt.subplots(figsize=(6.6, 4.0), dpi=180)
    ax.semilogy(valid_ranks, [errors[str(rank)] for rank in valid_ranks], "o-", linewidth=1.2)
    ax.set_xlabel("Truncation rank")
    ax.set_ylabel("Relative fluctuation reconstruction error")
    ax.set_title("POD low-rank reconstruction error")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return errors


def save_phase_portrait(path: Path, coefficients: np.ndarray) -> None:
    """Save the first two POD coefficients as a phase portrait."""

    if coefficients.shape[0] < 2:
        return

    fig, ax = plt.subplots(figsize=(4.8, 4.5), dpi=180)
    ax.plot(coefficients[0], coefficients[1], linewidth=0.7)
    ax.scatter(coefficients[0, 0], coefficients[1, 0], s=24, label="start")
    ax.set_xlabel("a1")
    ax.set_ylabel("a2")
    ax.set_title("POD coefficient phase portrait")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_lift_window_plot(
    path: Path,
    output: np.ndarray,
    snapshot_time: np.ndarray,
) -> None:
    """Save lift coefficient history and the analyzed snapshot time window."""

    if output.ndim != 2 or output.shape[1] < 2:
        return

    fig, ax = plt.subplots(figsize=(8.0, 3.8), dpi=180)
    ax.plot(output[:, 0], output[:, 1], linewidth=0.8, label="lift coefficient")
    ax.axvspan(snapshot_time[0], snapshot_time[-1], color="tab:orange", alpha=0.2, label="POD snapshots")
    ax.set_xlabel("time")
    ax.set_ylabel("Cl")
    ax.set_title("Lift coefficient and POD snapshot window")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def run_full_pod_analysis(
    config: PodConfig,
    figures_dir: Path,
    results_dir: Path,
) -> dict[str, object]:
    """Run full 2000-snapshot POD analysis and save figures/results."""

    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    data_dir = resolve_data_dir(config)
    print(f"Data directory: {data_dir}", flush=True)

    coordinates, nodes, pressure, output = load_oat15a_data(data_dir)
    print(f"coordinates: {coordinates.shape}", flush=True)
    print(f"nodes: {nodes.shape}", flush=True)
    print(f"pressure snapshots: {pressure.shape}", flush=True)
    print(f"output: {output.shape}", flush=True)

    snapshot_time = time_axis(pressure.shape[1], config.dt, config.snapshot_start_time)
    mean_pressure, fluctuation = pressure_fluctuation(pressure)
    del pressure

    print("Computing POD by snapshot method on the full snapshot matrix...", flush=True)
    modes, singular_values, coefficients, energy = snapshot_pod(fluctuation)
    print("POD computation finished.", flush=True)

    triangulation = make_triangulation(coordinates, nodes)
    fft_peaks = coefficient_fft_peaks(
        coefficients=coefficients,
        dt=config.dt,
        reduced_frequency_factor=config.reduced_frequency_factor,
        count=config.fft_coefficients,
    )

    print("Saving figures...", flush=True)
    save_energy_plot(figures_dir / "pod_energy_spectrum.png", energy, config.spectrum_modes)
    save_coefficients_plot(
        figures_dir / "pod_temporal_coefficients.png",
        snapshot_time,
        coefficients,
        count=config.fft_coefficients,
    )
    save_fft_plot(
        figures_dir / "pod_temporal_coefficients_fft.png",
        coefficients,
        dt=config.dt,
        reduced_frequency_factor=config.reduced_frequency_factor,
        count=config.fft_coefficients,
    )
    reconstruction_errors = save_reconstruction_error_plot(
        figures_dir / "pod_low_rank_reconstruction_error.png",
        energy,
        config.energy_ranks,
    )
    save_phase_portrait(figures_dir / "pod_phase_portrait_a1_a2.png", coefficients)
    save_lift_window_plot(figures_dir / "lift_coefficient_snapshot_window.png", output, snapshot_time)
    save_field_plot(
        figures_dir / "mean_pressure.png",
        triangulation,
        mean_pressure,
        "Mean pressure",
        cmap="viridis",
        symmetric=False,
    )
    save_field_plot(
        figures_dir / "mean_pressure_zoom.png",
        triangulation,
        mean_pressure,
        "Mean pressure, zoomed",
        cmap="viridis",
        symmetric=False,
        xlim=config.zoom_xlim,
        ylim=config.zoom_ylim,
    )

    modes_to_plot = min(config.modes_to_plot, modes.shape[1])
    for mode_index in range(modes_to_plot):
        save_field_plot(
            figures_dir / f"pod_mode_{mode_index + 1:02d}.png",
            triangulation,
            modes[:, mode_index],
            f"POD mode {mode_index + 1}",
        )
        save_field_plot(
            figures_dir / f"pod_mode_{mode_index + 1:02d}_zoom.png",
            triangulation,
            modes[:, mode_index],
            f"POD mode {mode_index + 1}, zoomed",
            xlim=config.zoom_xlim,
            ylim=config.zoom_ylim,
        )

    modes_to_save = min(config.modes_to_save, modes.shape[1])
    np.savez_compressed(
        results_dir / "oat15a_pod_results.npz",
        snapshot_time=snapshot_time,
        mean_pressure=mean_pressure,
        pod_energy=energy,
        pod_singular_values=singular_values,
        pod_coefficients=coefficients[:modes_to_save],
        pod_modes=modes[:, :modes_to_save],
        coordinates=coordinates,
        nodes=nodes,
    )

    cumulative = np.cumsum(energy)
    summary: dict[str, object] = {
        "data_dir": str(data_dir),
        "pressure_shape": [int(fluctuation.shape[0]), int(fluctuation.shape[1])],
        "snapshot_matrix": "pressure fluctuation, X = Snapshotp - temporal mean",
        "dt": float(config.dt),
        "snapshot_start_time": float(config.snapshot_start_time),
        "snapshot_end_time": float(snapshot_time[-1]),
        "reduced_frequency_factor": float(config.reduced_frequency_factor),
        "first_10_energy_fraction": [float(x) for x in energy[:10]],
        "cumulative_energy_rank_2": float(cumulative[1]),
        "cumulative_energy_rank_4": float(cumulative[3]),
        "cumulative_energy_rank_8": float(cumulative[7]),
        "cumulative_energy_rank_10": float(cumulative[9]),
        "cumulative_energy_rank_20": float(cumulative[19]),
        "low_rank_reconstruction_errors": reconstruction_errors,
        "pod_coefficient_fft_peaks": fft_peaks,
        "figures_dir": str(figures_dir),
        "results_file": str(results_dir / "oat15a_pod_results.npz"),
    }

    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (results_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    print(f"Saved figures to: {figures_dir}", flush=True)
    print(f"Saved results to: {results_dir}", flush=True)
    return summary
