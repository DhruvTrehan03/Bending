"""Biomimetic conductivity optimizers and touch sensitivity costs.

This module is intentionally standalone so it can be imported from the
existing BendingSim scripts without changing the current UI flow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _validate_positive_int(name: str, value: int) -> int:
    value_int = int(value)
    if value_int <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value_int


def _validate_positive_float(name: str, value: float) -> float:
    value_float = float(value)
    if value_float <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return value_float


def _stable_logsumexp(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float).ravel()
    if values.size == 0:
        return float("inf")
    max_value = float(np.max(values))
    if not np.isfinite(max_value):
        return max_value
    shifted = np.exp(values - max_value)
    return float(max_value + np.log(np.sum(shifted)))


@dataclass
class _CachedField:
    sigma: np.ndarray | None = None
    phi: np.ndarray | None = None
    current_density: np.ndarray | None = None
    cost: float | None = None


class FungalGrowthEIT:
    """Biased random-walk conductivity growth on a mesh adjacency graph."""

    def __init__(
        self,
        mesh: Any,
        sigma_0: float = 1.0,
        alpha: float = 1.5,
        rho: float = 0.99,
        n_agents: int = 10,
        n_steps: int = 100,
        fem_solve_every: int = 10,
        sigma_max: float | None = None,
        normalise: bool = False,
        rng: np.random.Generator | int | None = None,
    ) -> None:
        self.mesh = mesh
        self.sigma_0 = float(sigma_0)
        self.alpha = float(alpha)
        self.rho = float(rho)
        self.n_agents = _validate_positive_int("n_agents", n_agents)
        self.n_steps = _validate_positive_int("n_steps", n_steps)
        self.fem_solve_every = _validate_positive_int("fem_solve_every", fem_solve_every)
        self.sigma_max = None if sigma_max is None else float(sigma_max)
        if self.sigma_max is not None and self.sigma_max <= 0.0:
            raise ValueError("sigma_max must be positive when provided.")
        self.normalise = bool(normalise)
        self.rng = np.random.default_rng(rng)

        if not hasattr(self.mesh, "n_elements"):
            raise AttributeError("mesh must provide n_elements.")
        if not hasattr(self.mesh, "adjacent_elements"):
            raise AttributeError("mesh must provide adjacent_elements(elem_idx).")
        if not hasattr(self.mesh, "solve"):
            raise AttributeError("mesh must provide solve(sigma).")
        if not hasattr(self.mesh, "current_density"):
            raise AttributeError("mesh must provide current_density(phi).")
        if not hasattr(self.mesh, "gradient_sq"):
            raise AttributeError("mesh must provide gradient_sq(phi).")
        if not hasattr(self.mesh, "areas"):
            raise AttributeError("mesh must provide areas.")

        self.n_elements = int(self.mesh.n_elements)
        if self.n_elements <= 0:
            raise ValueError("mesh.n_elements must be positive.")

        self.reset()

    def reset(self) -> None:
        """Reset visit counters to 1 everywhere."""

        self.counters = np.ones(self.n_elements, dtype=float)
        self._cache = _CachedField()

    def conductivity(self) -> np.ndarray:
        """Return the current conductivity field."""

        sigma = self.sigma_0 * np.asarray(self.counters, dtype=float)
        if self.normalise:
            mean_sigma = float(np.mean(sigma))
            if mean_sigma > 0.0:
                sigma = sigma * (self.sigma_0 / mean_sigma)
        if self.sigma_max is not None:
            sigma = np.minimum(sigma, self.sigma_max)
        return np.asarray(sigma, dtype=float)

    def _solve_field(self) -> tuple[np.ndarray, np.ndarray]:
        sigma = self.conductivity()
        if self._cache.sigma is not None and np.array_equal(self._cache.sigma, sigma):
            if self._cache.phi is not None and self._cache.current_density is not None:
                return self._cache.phi, self._cache.current_density

        phi = np.asarray(self.mesh.solve(sigma), dtype=float)
        current_density = np.asarray(self.mesh.current_density(phi), dtype=float).ravel()
        if current_density.size != self.n_elements:
            raise ValueError(
                f"mesh.current_density(phi) returned size {current_density.size}, expected {self.n_elements}."
            )

        self._cache = _CachedField(sigma=sigma.copy(), phi=phi, current_density=current_density)
        return phi, current_density

    def _cost_from_sigma_and_phi(self, sigma: np.ndarray, phi: np.ndarray) -> float:
        grad_sq = np.asarray(self.mesh.gradient_sq(phi), dtype=float).ravel()
        areas = np.asarray(self.mesh.areas, dtype=float).ravel()
        if grad_sq.size != self.n_elements:
            raise ValueError(
                f"mesh.gradient_sq(phi) returned size {grad_sq.size}, expected {self.n_elements}."
            )
        if areas.size != self.n_elements:
            raise ValueError(f"mesh.areas size {areas.size}, expected {self.n_elements}.")

        cost = -float(np.sum(np.asarray(sigma, dtype=float) * grad_sq * areas))
        return cost

    def cost(self) -> float:
        """Return the current power-dissipation cost."""

        sigma = self.conductivity()
        phi, _current_density = self._solve_field()
        cost = self._cost_from_sigma_and_phi(sigma, phi)
        self._cache.cost = cost
        return cost

    def _pick_start_element(self) -> int:
        return int(self.rng.integers(0, self.n_elements))

    def _choose_next_element(self, current_element: int, current_density: np.ndarray) -> int:
        neighbors = list(self.mesh.adjacent_elements(int(current_element)))
        if not neighbors:
            return int(current_element)

        neighbor_indices = np.asarray(neighbors, dtype=int)
        weights = np.abs(np.asarray(current_density, dtype=float)[neighbor_indices]) ** self.alpha
        weights = np.asarray(weights, dtype=float)
        if not np.any(np.isfinite(weights)) or float(np.sum(weights)) <= 0.0:
            probabilities = np.full(neighbor_indices.size, 1.0 / neighbor_indices.size, dtype=float)
        else:
            weights = np.clip(weights, 0.0, np.inf)
            probabilities = weights / float(np.sum(weights))
        return int(self.rng.choice(neighbor_indices, p=probabilities))

    def _apply_evaporation(self) -> None:
        if self.rho == 1.0:
            return
        self.counters *= self.rho
        self._cache = _CachedField()

    def _agent_walk(self) -> None:
        phi, current_density = self._solve_field()
        current_element = self._pick_start_element()
        self.counters[current_element] += 1.0

        for step in range(self.n_steps):
            current_element = self._choose_next_element(current_element, current_density)
            self.counters[current_element] += 1.0

            if (step + 1) % self.fem_solve_every == 0:
                phi, current_density = self._solve_field()

        _ = phi  # Ensure the final field is retained for debugging and inspection.
        self._apply_evaporation()

    def grow(self, n_iterations: int) -> np.ndarray:
        """Run the fungal growth loop and return the cost history."""

        iterations = int(n_iterations)
        if iterations < 0:
            raise ValueError("n_iterations must be non-negative.")

        history = []
        for _ in range(iterations):
            for _agent in range(self.n_agents):
                self._agent_walk()
            history.append(self.cost())
        return np.asarray(history, dtype=float)


class TouchSensitivityCost:
    """Touch-patch sensitivity objectives for a circular touch target."""

    def __init__(
        self,
        J: np.ndarray,
        mesh_elements: np.ndarray,
        touch_radius: float,
        noise_cov: np.ndarray | None = None,
        rng: np.random.Generator | int | None = None,
        regularisation: float = 1e-8,
    ) -> None:
        self.J = np.asarray(J, dtype=float)
        self.mesh_elements = np.asarray(mesh_elements, dtype=float)
        self.touch_radius = _validate_positive_float("touch_radius", touch_radius)
        self.rng = np.random.default_rng(rng)
        self.regularisation = float(regularisation)
        if self.regularisation < 0.0:
            raise ValueError("regularisation must be non-negative.")

        if self.mesh_elements.ndim != 2 or self.mesh_elements.shape[1] != 2:
            raise ValueError("mesh_elements must have shape (n_elements, 2).")
        if self.J.ndim != 2:
            raise ValueError("J must have shape (n_measurements, n_elements).")
        if self.J.shape[1] != self.mesh_elements.shape[0]:
            raise ValueError(
                f"J has {self.J.shape[1]} elements but mesh_elements has {self.mesh_elements.shape[0]}."
            )

        self.n_measurements, self.n_elements = self.J.shape
        self._bbox_min = np.min(self.mesh_elements, axis=0)
        self._bbox_max = np.max(self.mesh_elements, axis=0)
        self._radius_sq = float(self.touch_radius**2)

        self.noise_cov = None
        self.noise_cov_inv = None
        if noise_cov is not None:
            cov = np.asarray(noise_cov, dtype=float)
            if cov.shape != (self.n_measurements, self.n_measurements):
                raise ValueError(
                    "noise_cov must have shape (n_measurements, n_measurements)."
                )
            self.noise_cov = cov
            self.noise_cov_inv = np.linalg.pinv(
                cov + self.regularisation * np.eye(self.n_measurements, dtype=float)
            )

    def _sample_patches(self, n_samples: int) -> list[np.ndarray]:
        """Sample valid circular touch patches within the mesh bounding box."""

        samples = _validate_positive_int("n_samples", n_samples)
        patches: list[np.ndarray] = []
        max_attempts = max(1000, samples * 100)
        attempts = 0

        while len(patches) < samples and attempts < max_attempts:
            attempts += 1
            point = self.rng.uniform(self._bbox_min, self._bbox_max)
            d2 = np.sum((self.mesh_elements - point) ** 2, axis=1)
            patch = np.flatnonzero(d2 <= self._radius_sq)
            if patch.size == 0:
                continue
            patches.append(patch.astype(int, copy=False))

        if len(patches) < samples:
            raise RuntimeError(
                f"Could not sample {samples} valid touch patches after {attempts} attempts."
            )
        return patches

    def _patch_signals(self, patches: list[np.ndarray]) -> np.ndarray:
        signals = []
        for patch in patches:
            signal = np.sum(self.J[:, patch], axis=1)
            signals.append(np.asarray(signal, dtype=float).ravel())
        return np.asarray(signals, dtype=float)

    def _patch_energies(self, n_samples: int) -> np.ndarray:
        patches = self._sample_patches(n_samples)
        signals = self._patch_signals(patches)
        return np.sum(signals**2, axis=1)

    def expected_sensitivity(self, n_samples: int = 200) -> float:
        """Return the negative mean patch sensitivity."""

        energies = self._patch_energies(n_samples)
        return float(-np.mean(energies))

    def minimax_sensitivity(self, n_samples: int = 200) -> float:
        """Return the negative worst-case patch sensitivity."""

        energies = self._patch_energies(n_samples)
        return float(-np.min(energies))

    def softmin_sensitivity(self, n_samples: int = 200, temperature: float = 1.0) -> float:
        """Smooth approximation of minimax sensitivity using a soft minimum."""

        temperature = _validate_positive_float("temperature", temperature)
        alpha = 1.0 / temperature
        energies = self._patch_energies(n_samples)
        softmin = (1.0 / alpha) * _stable_logsumexp(-alpha * energies)
        return float(softmin)

    def snr_sensitivity(self, n_samples: int = 200) -> float:
        """Return the negative mean SNR-style patch score."""

        patches = self._sample_patches(n_samples)
        signals = self._patch_signals(patches)
        if self.noise_cov_inv is None:
            snr_values = np.sum(signals**2, axis=1)
        else:
            snr_values = np.einsum("bi,ij,bj->b", signals, self.noise_cov_inv, signals)
        return float(-np.mean(snr_values))

    def distinguishability(self, n_samples: int = 100) -> float:
        """Return the negative mean pairwise patch distinguishability."""

        samples = _validate_positive_int("n_samples", n_samples)
        patches = self._sample_patches(samples * 2)
        signals = self._patch_signals(patches)
        diffs = signals[0::2] - signals[1::2]
        distinguishability_values = np.sum(diffs**2, axis=1)
        return float(-np.mean(distinguishability_values))

    def combined(self, n_samples: int = 200, lambda_uniformity: float = 0.1) -> float:
        """Combine mean sensitivity with a spatial uniformity penalty."""

        lambda_uniformity = float(lambda_uniformity)
        energies = self._patch_energies(n_samples)
        mean_term = float(-np.mean(energies))
        uniformity_term = float(np.var(energies))
        return float(mean_term + lambda_uniformity * uniformity_term)
