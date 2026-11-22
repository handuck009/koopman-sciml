"""Minimal Exact DMD utilities.

The functions in this file avoid any DMD-specific third-party package so the
linear algebra stays visible. They are meant to support the first notebook.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DMDResult:
    eigvals: np.ndarray
    modes: np.ndarray
    amplitudes: np.ndarray
    a_tilde: np.ndarray
    rank: int


def make_synthetic_flow(nx: int = 200, nt: int = 160) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Make a synthetic flow with two oscillatory modes and exponential decay."""
    x = np.linspace(-10.0, 10.0, nx)
    t = np.linspace(0.0, 10.0, nt)

    envelope_1 = np.exp(-((x + 3.0) ** 2) / 12.0)
    envelope_2 = np.exp(-((x - 2.0) ** 2) / 18.0)

    # Each oscillatory frequency needs a sine/cosine spatial pair. This gives
    # DMD a clean low-rank linear representation.
    phi_1c = envelope_1 * np.cos(0.9 * x)
    phi_1s = envelope_1 * np.sin(0.9 * x)
    phi_2c = envelope_2 * np.cos(1.7 * x)
    phi_2s = envelope_2 * np.sin(1.7 * x)

    omega_1 = 2.0
    omega_2 = 5.0
    sigma_1 = -0.04
    sigma_2 = -0.09

    data = np.zeros((nx, nt), dtype=float)
    for k, tk in enumerate(t):
        data[:, k] = (
            np.exp(sigma_1 * tk) * (phi_1c * np.cos(omega_1 * tk) + phi_1s * np.sin(omega_1 * tk))
            + 0.5
            * np.exp(sigma_2 * tk)
            * (phi_2c * np.cos(omega_2 * tk) + phi_2s * np.sin(omega_2 * tk))
        )

    return x, t, data

def make_lorenz(
    nt=5000,
    dt=0.01,
    sigma=10.0,
    rho=28.0,
    beta=8.0 / 3.0,
    x0=None,
):
    if x0 is None:
        x0 = np.array([1.0, 1.0, 1.0], dtype=float)

    def f(state):
        x, y, z = state
        return np.array([
            sigma * (y - x),
            x * (rho - z) - y,
            x * y - beta * z,
        ])

    data = np.zeros((3, nt), dtype=float)
    data[:, 0] = x0

    for k in range(nt - 1):
        # RK4 integration
        s = data[:, k]
        k1 = f(s)
        k2 = f(s + 0.5 * dt * k1)
        k3 = f(s + 0.5 * dt * k2)
        k4 = f(s + dt * k3)
        data[:, k + 1] = s + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    t = np.arange(nt) * dt
    return t, data

def snapshot_pair(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build X and Xp from time-ordered snapshots.

    X  = [x_1, x_2, ..., x_{m-1}]
    Xp = [x_2, x_3, ..., x_m]
    """
    if data.ndim != 2:
        raise ValueError("data must be a 2D array with snapshots stored as columns")
    if data.shape[1] < 2:
        raise ValueError("data must contain at least two snapshots")
    return data[:, :-1], data[:, 1:]


def exact_dmd(X: np.ndarray, Xp: np.ndarray, rank: int | None = None) -> DMDResult:

    if X.shape != Xp.shape:
        raise ValueError(f"X and Xp must have the same shape, got {X.shape} and {Xp.shape}")

    U, S, Vh = np.linalg.svd(X, full_matrices=False)

    if rank is None:
        rank = int(np.sum(S > np.finfo(float).eps * max(X.shape) * S[0]))
    if rank < 1 or rank > len(S):
        raise ValueError(f"rank must be in [1, {len(S)}], got {rank}")

    U_r = U[:, :rank]
    S_r = S[:rank]
    V_r = Vh.conj().T[:, :rank]

    S_inv = np.diag(1.0 / S_r)
    a_tilde = U_r.conj().T @ Xp @ V_r @ S_inv

    eigvals, W = np.linalg.eig(a_tilde)
    modes = Xp @ V_r @ S_inv @ W
    amplitudes = np.linalg.lstsq(modes, X[:, 0], rcond=None)[0]

    return DMDResult(
        eigvals=eigvals,
        modes=modes,
        amplitudes=amplitudes,
        a_tilde=a_tilde,
        rank=rank,
    )


def reconstruct(result: DMDResult, steps: int, x0: np.ndarray | None = None) -> np.ndarray:
    """Reconstruct or forecast snapshots using a fitted DMD model.

    If x0 is supplied, amplitudes are recomputed from that initial state.
    """
    if steps < 1:
        raise ValueError("steps must be positive")

    if x0 is None:
        amplitudes = result.amplitudes
    else:
        amplitudes = np.linalg.lstsq(result.modes, x0, rcond=None)[0]

    dynamics = np.zeros((result.rank, steps), dtype=complex)
    for k in range(steps):
        dynamics[:, k] = amplitudes * (result.eigvals**k)

    return (result.modes @ dynamics).real


def relative_error(true: np.ndarray, pred: np.ndarray) -> float:
    """Return ||true - pred||_F / ||true||_F."""
    denom = np.linalg.norm(true)
    if denom == 0.0:
        return float(np.linalg.norm(true - pred))
    return float(np.linalg.norm(true - pred) / denom)
