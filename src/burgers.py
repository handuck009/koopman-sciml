"""Finite-difference data generator for the 1D viscous Burgers equation.

The equation is

    u_t + u u_x = nu u_xx

This file only generates snapshot data. DMD is kept in ``dmd.py`` so the
project structure stays clear:

- burgers.py: physical system / CFD data
- dmd.py: data-driven modeling method
- notebooks/: experiments and plots
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def _burgers_rhs_periodic(u: np.ndarray, dx: float, nu: float) -> np.ndarray:
    """Return the right-hand side of viscous Burgers with periodic boundaries.

    The nonlinear flux uses a local Lax-Friedrichs/Rusanov flux. This is more
    robust than a plain central difference for the convection term.
    """
    u_right = np.roll(u, -1)
    u_left = np.roll(u, 1)

    flux = 0.5 * u**2
    flux_right = 0.5 * (flux + np.roll(flux, -1))
    flux_right -= 0.5 * np.maximum(np.abs(u), np.abs(u_right)) * (u_right - u)
    flux_left = np.roll(flux_right, 1)

    convection = -(flux_right - flux_left) / dx
    diffusion = nu * (u_right - 2.0 * u + u_left) / dx**2
    return convection + diffusion


def _rk4_step(u: np.ndarray, dt: float, dx: float, nu: float) -> np.ndarray:
    """Advance one time step with classical fourth-order Runge-Kutta."""
    k1 = _burgers_rhs_periodic(u, dx, nu)
    k2 = _burgers_rhs_periodic(u + 0.5 * dt * k1, dx, nu)
    k3 = _burgers_rhs_periodic(u + 0.5 * dt * k2, dx, nu)
    k4 = _burgers_rhs_periodic(u + dt * k3, dx, nu)
    return u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def make_burgers_1d(
    nx: int = 256,
    nt: int = 181,
    t_end: float = 1.5,
    nu: float = 0.03,
    domain: tuple[float, float] = (0.0, 2.0 * np.pi),
    cfl: float = 0.25,
    initial_condition: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate snapshots for the 1D viscous Burgers equation.

    Parameters
    ----------
    nx:
        Number of spatial grid points.
    nt:
        Number of saved time snapshots.
    t_end:
        Final saved time.
    nu:
        Viscosity. Larger values make the flow smoother.
    domain:
        Periodic spatial domain.
    cfl:
        Safety factor for adaptive internal time stepping.
    initial_condition:
        Optional function ``u0 = f(x)``. If omitted, a two-frequency sine wave
        is used.

    Returns
    -------
    x:
        Spatial grid with shape ``(nx,)``.
    t:
        Saved time points with shape ``(nt,)``.
    data:
        Snapshot matrix with shape ``(nx, nt)``. Each column is one state
        ``u(x, t_k)``.
    """
    if nx < 8:
        raise ValueError("nx must be at least 8")
    if nt < 2:
        raise ValueError("nt must be at least 2")
    if t_end <= 0.0:
        raise ValueError("t_end must be positive")
    if nu < 0.0:
        raise ValueError("nu must be non-negative")
    if not (0.0 < cfl <= 1.0):
        raise ValueError("cfl must be in (0, 1]")

    x0, x1 = domain
    if x1 <= x0:
        raise ValueError("domain must satisfy domain[1] > domain[0]")

    length = x1 - x0
    dx = length / nx
    x = np.linspace(x0, x1, nx, endpoint=False)
    t = np.linspace(0.0, t_end, nt)

    if initial_condition is None:
        u = np.sin(x) + 0.5 * np.sin(2.0 * x)
    else:
        u = np.asarray(initial_condition(x), dtype=float)

    if u.shape != (nx,):
        raise ValueError(f"initial_condition must return shape ({nx},), got {u.shape}")

    data = np.zeros((nx, nt), dtype=float)
    data[:, 0] = u

    current_time = 0.0
    for j in range(1, nt):
        target_time = t[j]
        while current_time < target_time - 1e-14:
            max_speed = max(float(np.max(np.abs(u))), 1e-12)
            dt_advective = dx / max_speed
            dt_diffusive = np.inf if nu == 0.0 else 0.5 * dx**2 / nu
            dt = cfl * min(dt_advective, dt_diffusive)
            dt = min(dt, target_time - current_time)

            u = _rk4_step(u, dt, dx, nu)
            current_time += dt

        data[:, j] = u

    return x, t, data
