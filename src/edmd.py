"""Minimal EDMD utilities for nonlinear dynamical systems.

Convention used in this project:

- State snapshots are stored as columns: ``data.shape == (n_states, n_steps)``.
- Lifted observables are also stored as columns: ``phi.shape == (n_features, n_steps)``.
- EDMD fits ``Phi_Y ~= K @ Phi_X``.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement

import numpy as np


@dataclass
class PolynomialDictionary:
    """Hand-built polynomial observables up to a chosen total degree."""

    n_states: int
    degree: int = 2
    include_bias: bool = True

    def __post_init__(self) -> None:
        if self.n_states < 1:
            raise ValueError("n_states must be positive")
        if self.degree < 1:
            raise ValueError("degree must be at least 1")

        powers: list[tuple[int, ...]] = []
        if self.include_bias:
            powers.append(())
        for total_degree in range(1, self.degree + 1):
            powers.extend(combinations_with_replacement(range(self.n_states), total_degree))
        self.powers = powers

    @property
    def n_features(self) -> int:
        return len(self.powers)

    def names(self, variables: list[str] | None = None) -> list[str]:
        """Return human-readable names for the dictionary terms."""
        if variables is None:
            variables = [f"x{i + 1}" for i in range(self.n_states)]
        if len(variables) != self.n_states:
            raise ValueError(f"expected {self.n_states} variable names, got {len(variables)}")

        names: list[str] = []
        for power in self.powers:
            if not power:
                names.append("1")
                continue
            counts = {idx: power.count(idx) for idx in sorted(set(power))}
            parts = []
            for idx, exponent in counts.items():
                if exponent == 1:
                    parts.append(variables[idx])
                else:
                    parts.append(f"{variables[idx]}^{exponent}")
            names.append("*".join(parts))
        return names

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Lift state snapshots into polynomial observable snapshots."""
        data = np.asarray(data, dtype=float)
        if data.ndim == 1:
            data = data.reshape(self.n_states, 1)
        if data.ndim != 2:
            raise ValueError("data must be a 1D state or a 2D snapshot matrix")
        if data.shape[0] != self.n_states:
            raise ValueError(f"expected {self.n_states} state variables, got {data.shape[0]}")

        features = np.empty((self.n_features, data.shape[1]), dtype=float)
        for row, power in enumerate(self.powers):
            if not power:
                features[row, :] = 1.0
            else:
                values = np.ones(data.shape[1], dtype=float)
                for idx in power:
                    values *= data[idx, :]
                features[row, :] = values
        return features


@dataclass
class EDMDResult:
    dictionary: PolynomialDictionary
    koopman_matrix: np.ndarray
    decoder: np.ndarray
    rcond: float


def fit_edmd(
    X: np.ndarray,
    Y: np.ndarray,
    dictionary: PolynomialDictionary,
    rcond: float = 1e-10,
) -> EDMDResult:
    """Fit an EDMD model from paired snapshots ``X`` and ``Y``.

    ``X`` and ``Y`` should have matching shapes ``(n_states, n_samples)`` and
    represent one-step pairs. The fitted model evolves lifted features and then
    decodes them back to the original state variables.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.shape != Y.shape:
        raise ValueError(f"X and Y must have the same shape, got {X.shape} and {Y.shape}")
    if X.ndim != 2:
        raise ValueError("X and Y must be 2D snapshot matrices")

    Phi_X = dictionary.transform(X)
    Phi_Y = dictionary.transform(Y)

    # Column convention: Phi_Y ~= K @ Phi_X.
    koopman_matrix = Phi_Y @ np.linalg.pinv(Phi_X, rcond=rcond)

    # Linear decoder: X ~= decoder @ Phi_X.
    decoder = X @ np.linalg.pinv(Phi_X, rcond=rcond)

    return EDMDResult(
        dictionary=dictionary,
        koopman_matrix=koopman_matrix,
        decoder=decoder,
        rcond=rcond,
    )


def edmd_rollout(result: EDMDResult, x0: np.ndarray, steps: int) -> np.ndarray:
    """Forecast original states by repeatedly evolving lifted observables."""
    if steps < 1:
        raise ValueError("steps must be positive")

    x0 = np.asarray(x0, dtype=float).reshape(result.dictionary.n_states, 1)
    phi = result.dictionary.transform(x0)

    pred = np.empty((result.decoder.shape[0], steps), dtype=float)
    for k in range(steps):
        pred[:, k] = (result.decoder @ phi).ravel()
        phi = result.koopman_matrix @ phi
    return pred


def edmd_one_step(result: EDMDResult, X: np.ndarray) -> np.ndarray:
    """Predict one step ahead from each supplied state snapshot."""
    Phi_X = result.dictionary.transform(X)
    Phi_Y_pred = result.koopman_matrix @ Phi_X
    return result.decoder @ Phi_Y_pred
