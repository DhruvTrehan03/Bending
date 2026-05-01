"""Simplified mesh generation and DE optimization UI for BendingSim.

Focus: uniform sensitivity (jacobian-based) with individual element conductivity states.

This app provides:
- mesh_generation.py: mesh creation (direct element-wise conductivity, no field parameterization)
- benchmarking.py: sensitivity uniformity objective and scoring
- optimization.py: DE-only optimization with generation callbacks

Workflow:
1. Generate a mesh.
2. Configure metric, model, and strategy sections.
3. Run optimization with live mesh + sensitivity updates.
4. Save full artifacts (mesh, conductivity, sensitivity, history).
5. Benchmark saved results.
"""

from __future__ import annotations

import csv
import json
import os
import queue
import traceback
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from scipy.optimize import differential_evolution
from tkinter import filedialog, messagebox, ttk
from pyeit.eit.fem import EITForward
import pyeit.eit.protocol as protocol
import pyeit.mesh as mesh
from pyeit.mesh import PyEITMesh
from pyeit.mesh.shape import rectangle


def timestamp_run_name(prefix: str = "ui") -> str:
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}"


def _coerce_value(value: Any) -> Any:
    if hasattr(value, "get"):
        return value.get()
    return value


def parse_float(value: Any, name: str) -> float:
    try:
        return float(_coerce_value(value))
    except ValueError as exc:
        raise ValueError(f"Invalid {name}: {_coerce_value(value)}") from exc


def parse_int(value: Any, name: str) -> int:
    try:
        return int(_coerce_value(value))
    except ValueError as exc:
        raise ValueError(f"Invalid {name}: {_coerce_value(value)}") from exc


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

        self.n_elements = int(self.mesh.n_elements)
        if self.n_elements <= 0:
            raise ValueError("mesh.n_elements must be positive.")

        self.reset()

    def reset(self) -> None:
        self.counters = np.ones(self.n_elements, dtype=float)
        self._cache = _CachedField()

    def conductivity(self) -> np.ndarray:
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
        self._cache = _CachedField(sigma=sigma.copy(), phi=phi, current_density=current_density)
        return phi, current_density

    def _cost_from_sigma_and_phi(self, sigma: np.ndarray, phi: np.ndarray) -> float:
        grad_sq = np.asarray(self.mesh.gradient_sq(phi), dtype=float).ravel()
        areas = np.asarray(self.mesh.areas, dtype=float).ravel()
        return -float(np.sum(np.asarray(sigma, dtype=float) * grad_sq * areas))

    def cost(self) -> float:
        sigma = self.conductivity()
        phi, _ = self._solve_field()
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

        _ = phi
        self._apply_evaporation()

    def grow(self, n_iterations: int) -> np.ndarray:
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

        self.n_measurements, self.n_elements = self.J.shape
        self._bbox_min = np.min(self.mesh_elements, axis=0)
        self._bbox_max = np.max(self.mesh_elements, axis=0)
        self._radius_sq = float(self.touch_radius ** 2)

        self.noise_cov_inv = None
        if noise_cov is not None:
            cov = np.asarray(noise_cov, dtype=float)
            self.noise_cov_inv = np.linalg.pinv(cov + self.regularisation * np.eye(self.n_measurements, dtype=float))

    def _sample_patches(self, n_samples: int) -> list[np.ndarray]:
        samples = _validate_positive_int("n_samples", n_samples)
        patches: list[np.ndarray] = []
        max_attempts = max(1000, samples * 100)
        attempts = 0

        while len(patches) < samples and attempts < max_attempts:
            attempts += 1
            point = self.rng.uniform(self._bbox_min, self._bbox_max)
            d2 = np.sum((self.mesh_elements - point) ** 2, axis=1)
            patch = np.flatnonzero(d2 <= self._radius_sq)
            if patch.size:
                patches.append(patch.astype(int, copy=False))

        if len(patches) < samples:
            raise RuntimeError(f"Could not sample {samples} valid touch patches after {attempts} attempts.")
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
        return np.sum(signals ** 2, axis=1)

    def expected_sensitivity(self, n_samples: int = 200) -> float:
        return float(-np.mean(self._patch_energies(n_samples)))

    def minimax_sensitivity(self, n_samples: int = 200) -> float:
        return float(-np.min(self._patch_energies(n_samples)))

    def softmin_sensitivity(self, n_samples: int = 200, temperature: float = 1.0) -> float:
        temperature = _validate_positive_float("temperature", temperature)
        alpha = 1.0 / temperature
        energies = self._patch_energies(n_samples)
        return float((1.0 / alpha) * _stable_logsumexp(-alpha * energies))

    def snr_sensitivity(self, n_samples: int = 200) -> float:
        patches = self._sample_patches(n_samples)
        signals = self._patch_signals(patches)
        if self.noise_cov_inv is None:
            snr_values = np.sum(signals ** 2, axis=1)
        else:
            snr_values = np.einsum("bi,ij,bj->b", signals, self.noise_cov_inv, signals)
        return float(-np.mean(snr_values))

    def distinguishability(self, n_samples: int = 100) -> float:
        samples = _validate_positive_int("n_samples", n_samples)
        patches = self._sample_patches(samples * 2)
        signals = self._patch_signals(patches)
        diffs = signals[0::2] - signals[1::2]
        return float(-np.mean(np.sum(diffs ** 2, axis=1)))

    def combined(self, n_samples: int = 200, lambda_uniformity: float = 0.1) -> float:
        energies = self._patch_energies(n_samples)
        return float(-np.mean(energies) + float(lambda_uniformity) * np.var(energies))


def rect_electrode_positions(n_el: int, p1: list[float], p2: list[float]) -> np.ndarray:
    if n_el % 4 != 0:
        raise ValueError("n_el must be divisible by 4")

    x1, y1 = p1
    x2, y2 = p2
    w, h = x2 - x1, y2 - y1
    perimeter = 2.0 * (w + h)
    spacing = perimeter / n_el

    pts = []
    for i in range(n_el):
        s = i * spacing
        if s < w:
            pts.append([x1 + s, y1])
        elif s < w + h:
            pts.append([x2, y1 + (s - w)])
        elif s < 2 * w + h:
            pts.append([x2 - (s - w - h), y2])
        else:
            pts.append([x1, y2 - (s - 2 * w - h)])
    return np.asarray(pts)


def build_model(n_el: int = 16, h0: float = 0.1, p1: list[float] | None = None, p2: list[float] | None = None):
    if p1 is None:
        p1 = [0.0, 0.0]
    if p2 is None:
        p2 = [5.0, 2.0]

    p_fix = rect_electrode_positions(n_el, p1, p2)
    mesh_obj = mesh.create(
        n_el=n_el,
        h0=h0,
        fd=lambda pts: rectangle(pts, p1=p1, p2=p2),
        bbox=[p1, p2],
        p_fix=p_fix,
    )

    protocol_obj = protocol.create(
        n_el=n_el,
        dist_exc=n_el // 2,
        step_meas=1,
        parser_meas="rotate_meas",
    )

    tri = np.asarray(mesh_obj.element, dtype=int)
    variable_mask = np.ones(tri.shape[0], dtype=bool)
    return mesh_obj, protocol_obj, variable_mask, p1, p2


def build_element_adjacency(mesh_obj: Any) -> list[list[int]]:
    tri = np.asarray(mesh_obj.element, dtype=int)
    n_elem = tri.shape[0]
    adjacency = [[] for _ in range(n_elem)]
    for i in range(n_elem):
        vertices_i = set(tri[i].tolist())
        for j in range(i + 1, n_elem):
            vertices_j = set(tri[j].tolist())
            if len(vertices_i & vertices_j) >= 2:
                adjacency[i].append(j)
                adjacency[j].append(i)
    return adjacency


def build_electrode_support_elements(mesh_obj: Any, variable_mask: np.ndarray, k_nearest: int = 4) -> list[np.ndarray]:
    tri = np.asarray(mesh_obj.element, dtype=int)
    pts = np.asarray(mesh_obj.node, dtype=float)
    centroids = np.mean(pts[tri], axis=1)[:, :2]
    variable_indices = np.where(np.asarray(variable_mask, dtype=bool))[0]
    variable_centroids = centroids[variable_indices]

    supports = []
    electrode_points = pts[np.asarray(mesh_obj.el_pos, dtype=int)][:, :2]
    for electrode_point in electrode_points:
        d2 = np.sum((variable_centroids - electrode_point) ** 2, axis=1)
        nearest_var_idx = np.argsort(d2)[:max(1, int(k_nearest))]
        supports.append(variable_indices[nearest_var_idx].astype(int))
    return supports


def branching_parameter_vector_size(field_shape_cfg: dict[str, Any]) -> int:
    max_children = int(max(1, field_shape_cfg.get("branching_max_children", 2)))
    depth_max = int(max(0, field_shape_cfg.get("branching_depth_max", 5)))
    node_count = 0
    nodes_this_level = 1
    for _ in range(depth_max + 1):
        node_count += nodes_this_level
        nodes_this_level *= max_children
    return 3 * node_count


def parameter_vector_size_from_cfg(field_shape_cfg: dict[str, Any], field_geometry: dict[str, Any] | None = None) -> int:
    model = str(field_shape_cfg.get("model", "element")).strip().lower()
    if model in {"branching", "fungal"}:
        return branching_parameter_vector_size(field_shape_cfg)
    if model == "element":
        if field_geometry is not None and "n_variable_elements" in field_geometry:
            return int(field_geometry["n_variable_elements"])
        return 0
    raise ValueError(f"Unsupported field-shape model: {model}")


def _electrode_array(electrode_points: Any) -> np.ndarray:
    if electrode_points is None:
        return np.empty((0, 2), dtype=float)
    arr = np.asarray(electrode_points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        return np.empty((0, 2), dtype=float)
    return arr


def _branching_segment_distance(xn: np.ndarray, yn: np.ndarray, x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    vx = x1 - x0
    vy = y1 - y0
    denom = vx * vx + vy * vy + np.finfo(float).tiny
    wx = xn - x0
    wy = yn - y0
    t = np.clip((wx * vx + wy * vy) / denom, 0.0, 1.0)
    px = x0 + t * vx
    py = y0 + t * vy
    return np.sqrt((xn - px) ** 2 + (yn - py) ** 2)


def _branching_tree_node_count(field_shape_cfg: dict[str, Any]) -> int:
    max_children = int(max(1, field_shape_cfg.get("branching_max_children", 2)))
    depth_max = int(max(0, field_shape_cfg.get("branching_depth_max", 5)))
    node_count = 0
    nodes_this_level = 1
    for _ in range(depth_max + 1):
        node_count += nodes_this_level
        nodes_this_level *= max_children
    return node_count


def _branching_tree_child_index(node_index: int, child_slot: int, max_children: int) -> int:
    return node_index * max_children + child_slot + 1


def _branching_tree_params_from_cfg(field_shape_cfg: dict[str, Any], theta: np.ndarray) -> dict[str, Any]:
    theta = np.asarray(theta, dtype=float).ravel()
    node_count = _branching_tree_node_count(field_shape_cfg)
    expected = 3 * node_count
    if theta.size != expected:
        raise ValueError(f"Expected {expected} parameters, got {theta.size}")

    node_params = np.clip(theta.reshape(node_count, 3), -1.0, 1.0)
    max_children = int(max(1, field_shape_cfg.get("branching_max_children", 2)))
    depth_max = int(max(0, field_shape_cfg.get("branching_depth_max", 5)))

    return {
        "node_params": node_params,
        "node_count": node_count,
        "max_children": max_children,
        "depth_max": depth_max,
        "angle_frac": float(field_shape_cfg.get("branching_angle_frac", 0.34)),
        "child_angle_frac": float(field_shape_cfg.get("branching_child_angle_frac", 0.34)),
        "size_min_frac": float(field_shape_cfg.get("branching_size_min_frac", 0.55)),
        "size_max_frac": float(field_shape_cfg.get("branching_size_max_frac", 1.0)),
        "root_length_frac": float(field_shape_cfg.get("branching_root_length_frac", 1.05)),
        "length_decay": float(np.clip(field_shape_cfg.get("branching_length_decay", 0.68), 0.35, 0.98)),
        "width_frac": float(field_shape_cfg.get("branching_width_frac", 0.12)),
        "width_decay": float(np.clip(field_shape_cfg.get("branching_width_decay", 0.80), 0.35, 0.98)),
        "seed_all_electrodes": bool(field_shape_cfg.get("branching_seed_all_electrodes", True)),
        "center_x": float(field_shape_cfg.get("branching_meet_x_frac", 0.0)),
        "center_y": float(field_shape_cfg.get("branching_meet_y_frac", 0.0)),
        "root_x": float(field_shape_cfg.get("branching_root_x_frac", 0.18)),
        "root_y": float(field_shape_cfg.get("branching_root_y_frac", 0.0)),
        "electrode_points_all": _electrode_array(field_shape_cfg.get("branching_seed_points_norm")),
        "electrode_points": _electrode_array(field_shape_cfg.get("branching_seed_points_norm_right")),
    }


def _evaluate_branching_tree_field_from_normalized_coords(
    xn: np.ndarray,
    yn: np.ndarray,
    theta: np.ndarray,
    field_shape_cfg: dict[str, Any],
) -> np.ndarray:
    params = _branching_tree_params_from_cfg(field_shape_cfg, theta)

    node_params = params["node_params"]
    node_count = params["node_count"]
    max_children = params["max_children"]
    depth_max = params["depth_max"]
    center_target = np.array([params["center_x"], params["center_y"]], dtype=float)

    if params["seed_all_electrodes"] and params["electrode_points_all"].size > 0:
        root_points = params["electrode_points_all"]
    elif params["electrode_points"].size > 0:
        root_points = params["electrode_points"]
    else:
        root_points = np.asarray([[params["root_x"], params["root_y"]]], dtype=float)

    segments: list[tuple[float, float, float, float, int]] = []

    def grow(node_index: int, x0: float, y0: float, base_angle: float, base_length: float, depth: int) -> None:
        if node_index >= node_count or depth > depth_max:
            return

        angle_raw, size_raw, child_raw = node_params[node_index]
        node_angle = base_angle + (np.pi * params["angle_frac"] * float(angle_raw))
        size_scale = params["size_min_frac"] + 0.5 * (float(size_raw) + 1.0) * (params["size_max_frac"] - params["size_min_frac"])
        segment_length = base_length * max(1e-6, size_scale)

        x1 = x0 + segment_length * np.cos(node_angle)
        y1 = y0 + segment_length * np.sin(node_angle)
        segments.append((x0, y0, x1, y1, depth))

        if depth >= depth_max or segment_length <= 1e-6:
            return

        child_count = int(np.clip(np.rint(0.5 * (float(child_raw) + 1.0) * max_children), 0, max_children))
        next_length = segment_length * params["length_decay"]
        child_angle_step = np.pi * params["child_angle_frac"]

        for child_slot in range(child_count):
            child_index = _branching_tree_child_index(node_index, child_slot, max_children)
            if child_index >= node_count:
                continue
            offset = child_slot - 0.5 * (child_count - 1)
            child_base_angle = node_angle + (offset * child_angle_step)
            grow(child_index, x1, y1, child_base_angle, next_length, depth + 1)

    for rx, ry in np.asarray(root_points, dtype=float):
        root_angle = float(np.arctan2(center_target[1] - ry, center_target[0] - rx))
        grow(0, float(rx), float(ry), root_angle, float(params["root_length_frac"]), 0)

    if not segments:
        return -0.5 * np.ones_like(xn, dtype=float)

    strength = np.zeros_like(xn, dtype=float)
    for x0, y0, x1, y1, depth in segments:
        dist = _branching_segment_distance(xn, yn, x0, y0, x1, y1)
        width = max(0.015, params["width_frac"] * (params["width_decay"] ** depth))
        depth_weight = params["length_decay"] ** depth
        contribution = depth_weight * np.exp(-0.5 * (dist / width) ** 2)
        strength = np.maximum(strength, contribution)

    if np.max(strength) > 0.0:
        strength = strength / np.max(strength)
    return strength - 0.5


def build_branching_field_geometry(mesh_obj: Any, variable_mask: np.ndarray, p1: list[float], p2: list[float]) -> dict[str, Any]:
    tri = np.asarray(mesh_obj.element, dtype=int)
    pts = np.asarray(mesh_obj.node, dtype=float)
    variable_indices = np.where(np.asarray(variable_mask, dtype=bool))[0]
    variable_centroids = np.mean(pts[tri], axis=1)[variable_indices, :2]

    x1, y1 = p1
    x2, y2 = p2
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    norm_scale = 0.5 * min(x2 - x1, y2 - y1)

    x_norm = (variable_centroids[:, 0] - cx) / norm_scale
    y_norm = (variable_centroids[:, 1] - cy) / norm_scale

    electrode_points = pts[np.asarray(mesh_obj.el_pos, dtype=int)][:, :2]
    electrode_points_norm = (electrode_points - np.array([cx, cy], dtype=float)) / norm_scale
    electrode_points_norm_right = electrode_points_norm[electrode_points_norm[:, 0] >= 0.0]

    return {
        "variable_indices": variable_indices,
        "centroids": np.mean(pts[tri], axis=1)[:, :2],
        "x_norm": x_norm,
        "y_norm": y_norm,
        "electrode_points_norm": electrode_points_norm,
        "electrode_points_norm_right": electrode_points_norm_right,
    }


def build_field_geometry(mesh_obj: Any, variable_mask: np.ndarray, p1: list[float], p2: list[float], field_shape_cfg: dict[str, Any]) -> dict[str, Any]:
    model = str(field_shape_cfg.get("model", "element")).strip().lower()
    if model in {"branching", "fungal"}:
        return build_branching_field_geometry(mesh_obj, variable_mask, p1, p2)
    if model == "element":
        tri = np.asarray(mesh_obj.element, dtype=int)
        pts = np.asarray(mesh_obj.node, dtype=float)
        return {
            "n_variable_elements": int(np.sum(variable_mask)),
            "centroids": np.mean(pts[tri], axis=1)[:, :2],
        }
    raise ValueError(f"Unsupported field-shape model: {model}")


def parameterized_state_from_theta(theta: np.ndarray, field_geometry: dict[str, Any], field_shape_cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    model = str(field_shape_cfg.get("model", "element")).strip().lower()
    theta = np.asarray(theta, dtype=float).ravel()
    expected = parameter_vector_size_from_cfg(field_shape_cfg, field_geometry)
    if theta.size != expected:
        raise ValueError(f"Expected {expected} parameters, got {theta.size}")

    if model in {"branching", "fungal"}:
        field = _evaluate_branching_tree_field_from_normalized_coords(
            xn=field_geometry["x_norm"],
            yn=field_geometry["y_norm"],
            theta=theta,
            field_shape_cfg=field_shape_cfg,
        )
        return np.asarray(field >= 0.0, dtype=bool), np.asarray(field, dtype=float)

    params_clipped = np.clip(theta, -1.0, 1.0)
    return np.asarray(params_clipped >= 0.0, dtype=bool), params_clipped


def expand_state_to_full_mesh(state_high_var: np.ndarray, variable_mask: np.ndarray) -> np.ndarray:
    state_high_var = np.asarray(state_high_var, dtype=bool)
    variable_mask = np.asarray(variable_mask, dtype=bool)
    state_full = np.zeros(variable_mask.size, dtype=bool)
    state_full[np.where(variable_mask)[0]] = state_high_var
    return state_full


def get_state_for_connectivity(state_high: np.ndarray, variable_mask: np.ndarray) -> np.ndarray:
    state_high = np.asarray(state_high, dtype=bool)
    variable_mask = np.asarray(variable_mask, dtype=bool)
    n_total = variable_mask.size
    n_var = int(np.sum(variable_mask))

    if state_high.size == n_total:
        return state_high
    if state_high.size == n_var:
        return expand_state_to_full_mesh(state_high, variable_mask)
    raise ValueError(f"state_high size {state_high.size} does not match {n_total} or {n_var}")


def make_permittivity(state_high: np.ndarray, variable_mask: np.ndarray, low_cond: float, high_cond: float) -> np.ndarray:
    n_elem = int(np.asarray(variable_mask, dtype=bool).size)
    perm = np.full(n_elem, float(low_cond), dtype=float)
    perm[np.asarray(state_high, dtype=bool)] = float(high_cond)
    return perm


def element_sensitivity_from_jacobian(jacobian: np.ndarray) -> np.ndarray:
    """Compute per-element sensitivity as sum of absolute Jacobian values.
    
    Uses raw sensitivity (not log-scaled) to match benchmarking.py calculations
    and provide better visualization of actual element sensitivities.
    
    Args:
        jacobian: Array of shape (n_measurements, n_elements)
    
    Returns:
        Per-element sensitivity: Array of shape (n_elements,)
    """
    return np.sum(np.abs(np.asarray(jacobian, dtype=float)), axis=0)


def sensitivity_for_display(sensitivity: np.ndarray) -> np.ndarray:
    """Map raw sensitivity to log-scale for stable visual comparison across runs."""
    vals = np.asarray(sensitivity, dtype=float).ravel()
    return np.log10(vals + np.finfo(float).tiny)


def compute_reference_sensitivity(mesh_obj: Any, protocol_obj: Any) -> np.ndarray:
    """Compute baseline sensitivity from a uniform conductivity field (test-script path)."""
    n_elem = int(np.asarray(mesh_obj.element).shape[0])
    perm_uniform = np.ones(n_elem, dtype=float)
    fwd = EITForward(mesh_obj, protocol_obj)
    jacobian, _ = fwd.compute_jac(perm=perm_uniform)
    return element_sensitivity_from_jacobian(jacobian)


def get_connected_components(state_high: np.ndarray, adjacency: list[list[int]]) -> list[list[int]]:
    state_high = np.asarray(state_high, dtype=bool)
    visited = np.zeros(state_high.size, dtype=bool)
    components = []
    for start in range(state_high.size):
        if not state_high[start] or visited[start]:
            continue
        queue_local = [start]
        visited[start] = True
        component = []
        while queue_local:
            elem = queue_local.pop(0)
            component.append(elem)
            for nxt in adjacency[elem]:
                if state_high[nxt] and not visited[nxt]:
                    visited[nxt] = True
                    queue_local.append(nxt)
        components.append(component)
    return components


def count_isolated_high_elements(state_high: np.ndarray, adjacency: list[list[int]]) -> int:
    state_high = np.asarray(state_high, dtype=bool)
    isolated = 0
    for i, is_high in enumerate(state_high):
        if is_high and not any(state_high[j] for j in adjacency[i]):
            isolated += 1
    return isolated


def has_connected_high_elements(state_high: np.ndarray, adjacency: list[list[int]]) -> bool:
    state_high = np.asarray(state_high, dtype=bool)
    n_high = int(np.sum(state_high))
    if n_high <= 1:
        return True
    return len(get_connected_components(state_high, adjacency)) == 1


def electrodes_connected_by_high_region(state_high: np.ndarray, adjacency: list[list[int]], electrode_supports: list[np.ndarray] | None) -> bool:
    if electrode_supports is None:
        return has_connected_high_elements(state_high, adjacency)

    if len(get_connected_components(state_high, adjacency)) != 1:
        return False

    for support in electrode_supports:
        support_idx = np.asarray(support, dtype=int)
        if support_idx.size == 0 or not np.any(np.asarray(state_high, dtype=bool)[support_idx]):
            return False
    return True


def entropy_score(sensitivity: np.ndarray) -> float:
    vals = np.abs(np.asarray(sensitivity, dtype=float).ravel())
    total = np.sum(vals)
    if total <= np.finfo(float).eps:
        return 0.0
    probabilities = vals / total
    nz = probabilities > np.finfo(float).eps
    return float(-np.sum(probabilities[nz] * np.log(probabilities[nz])) + np.mean(vals))


def _safe_uniformity(values: np.ndarray) -> float:
    vals = np.abs(np.asarray(values, dtype=float).ravel())
    mean_val = float(np.mean(vals))
    if mean_val <= np.finfo(float).eps:
        return 0.0
    cv = float(np.std(vals) / (mean_val + np.finfo(float).eps))
    return float(1.0 / (1.0 + cv))


BenchmarkRegistry: dict[str, Any] = {}


def register_benchmark(name: str):
    def decorator(fn):
        BenchmarkRegistry[str(name).strip().lower()] = fn
        return fn
    return decorator


def get_benchmark(name: str):
    key = str(name).strip().lower()
    if key not in BenchmarkRegistry:
        raise ValueError(f"Unknown benchmark '{name}'.")
    return BenchmarkRegistry[key]


@register_benchmark("uniformity")
def benchmark_uniformity(sensitivity: np.ndarray, state: np.ndarray, adjacency: list[list[int]] | None = None, weights: dict[str, float] | None = None, **_ignored) -> float:
    local_weights = {
        "entropy_weight": 0.5,
        "uniformity_weight": 1.0,
        "isolated_penalty": 2.0,
        "disconnected_penalty": 20.0,
    }
    if weights:
        local_weights.update(weights)

    score = -(
        float(local_weights["entropy_weight"]) * entropy_score(sensitivity)
        + float(local_weights["uniformity_weight"]) * _safe_uniformity(sensitivity)
    )

    if adjacency is not None:
        score += float(local_weights["isolated_penalty"]) * float(count_isolated_high_elements(state, adjacency))
        if not has_connected_high_elements(state, adjacency):
            score += float(local_weights["disconnected_penalty"])
    return float(score)


def _touch_metric_score(metric_name: str, weights: dict[str, float] | None, jacobian: np.ndarray, mesh_centroids: np.ndarray) -> float:
    local_weights = dict(weights or {})
    touch_radius = float(local_weights.get("touch_radius", 0.30))
    touch_samples = int(max(1, local_weights.get("touch_samples", 200)))
    touch_temperature = float(local_weights.get("touch_temperature", 1.0))
    touch_lambda = float(local_weights.get("touch_lambda", 0.1))
    touch_noise_scale = local_weights.get("touch_noise_scale", None)

    noise_cov = None
    if touch_noise_scale is not None and float(touch_noise_scale) > 0.0:
        noise_cov = np.eye(int(np.asarray(jacobian).shape[0]), dtype=float) * (float(touch_noise_scale) ** 2)

    touch_cost = TouchSensitivityCost(
        J=np.asarray(jacobian, dtype=float),
        mesh_elements=np.asarray(mesh_centroids, dtype=float),
        touch_radius=touch_radius,
        noise_cov=noise_cov,
    )

    metric = metric_name.strip().lower()
    if metric == "expected_sensitivity":
        return float(touch_cost.expected_sensitivity(touch_samples))
    if metric == "minimax_sensitivity":
        return float(touch_cost.minimax_sensitivity(touch_samples))
    if metric == "softmin_sensitivity":
        return float(touch_cost.softmin_sensitivity(touch_samples, temperature=touch_temperature))
    if metric == "snr_sensitivity":
        return float(touch_cost.snr_sensitivity(touch_samples))
    if metric == "distinguishability":
        return float(touch_cost.distinguishability(max(10, touch_samples // 2)))
    if metric == "combined":
        return float(touch_cost.combined(touch_samples, lambda_uniformity=touch_lambda))
    raise ValueError(f"Unsupported touch sensitivity benchmark '{metric_name}'.")


@register_benchmark("expected_sensitivity")
def benchmark_expected_sensitivity(sensitivity, state, adjacency=None, weights=None, jacobian=None, mesh_centroids=None, **_ignored):
    return _touch_metric_score("expected_sensitivity", weights, jacobian, mesh_centroids)


@register_benchmark("minimax_sensitivity")
def benchmark_minimax_sensitivity(sensitivity, state, adjacency=None, weights=None, jacobian=None, mesh_centroids=None, **_ignored):
    return _touch_metric_score("minimax_sensitivity", weights, jacobian, mesh_centroids)


@register_benchmark("softmin_sensitivity")
def benchmark_softmin_sensitivity(sensitivity, state, adjacency=None, weights=None, jacobian=None, mesh_centroids=None, **_ignored):
    return _touch_metric_score("softmin_sensitivity", weights, jacobian, mesh_centroids)


@register_benchmark("snr_sensitivity")
def benchmark_snr_sensitivity(sensitivity, state, adjacency=None, weights=None, jacobian=None, mesh_centroids=None, **_ignored):
    return _touch_metric_score("snr_sensitivity", weights, jacobian, mesh_centroids)


@register_benchmark("distinguishability")
def benchmark_distinguishability(sensitivity, state, adjacency=None, weights=None, jacobian=None, mesh_centroids=None, **_ignored):
    return _touch_metric_score("distinguishability", weights, jacobian, mesh_centroids)


@register_benchmark("combined")
def benchmark_combined(sensitivity, state, adjacency=None, weights=None, jacobian=None, mesh_centroids=None, **_ignored):
    return _touch_metric_score("combined", weights, jacobian, mesh_centroids)


def evaluate_state(
    fwd: EITForward,
    state: np.ndarray,
    variable_mask: np.ndarray,
    low_cond: float,
    high_cond: float,
    adjacency: list[list[int]] | None = None,
    electrode_supports: list[np.ndarray] | None = None,
    enforce_connectivity: bool = True,
    repair_disconnected: bool = False,
    benchmark_fn: Any = None,
    benchmark_name: str | None = None,
    benchmark_weights: dict[str, float] | None = None,
    mesh_centroids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    state_full = get_state_for_connectivity(state, variable_mask)

    if enforce_connectivity and not repair_disconnected and adjacency is not None:
        if not electrodes_connected_by_high_region(state_full, adjacency, electrode_supports):
            perm_bad = make_permittivity(state_full, variable_mask, low_cond, high_cond)
            return perm_bad, np.zeros_like(perm_bad, dtype=float), np.inf

    perm = make_permittivity(state_full, variable_mask, low_cond, high_cond)
    jacobian, _ = fwd.compute_jac(perm=perm)
    sensitivity = element_sensitivity_from_jacobian(jacobian)

    if benchmark_fn is None:
        benchmark_fn = get_benchmark(benchmark_name or "uniformity")

    score = float(
        benchmark_fn(
            sensitivity=sensitivity,
            state=state_full,
            adjacency=adjacency,
            weights=benchmark_weights,
            jacobian=np.asarray(jacobian, dtype=float),
            mesh_centroids=mesh_centroids,
        )
    )
    return perm, sensitivity, score


def optimize_parameterized_field_de(
    fwd: EITForward,
    variable_mask: np.ndarray,
    field_geometry: dict[str, Any],
    field_shape_cfg: dict[str, Any],
    low_cond: float,
    high_cond: float,
    maxiter: int,
    popsize: int,
    mutation: float | tuple[float, float],
    recombination: float,
    seed: int,
    adjacency: list[list[int]] | None = None,
    electrode_supports: list[np.ndarray] | None = None,
    enforce_connectivity: bool = True,
    repair_disconnected: bool = False,
    benchmark_fn: Any = None,
    benchmark_name: str | None = None,
    benchmark_weights: dict[str, float] | None = None,
    progress_callback: Any = None,
) -> dict[str, Any]:
    model = str(field_shape_cfg.get("model", "element")).strip().lower()
    mesh_centroids = field_geometry.get("centroids") if isinstance(field_geometry, dict) else None
    n_params = int(np.count_nonzero(variable_mask)) if model == "element" else parameter_vector_size_from_cfg(field_shape_cfg, field_geometry)
    if n_params == 0:
        raise ValueError("No parameters available for optimization.")

    maxiter = int(max(1, maxiter))
    popsize = int(max(2, popsize))
    recombination = float(np.clip(recombination, 0.0, 1.0))
    if np.isscalar(mutation):
        mutation = float(np.clip(mutation, 0.0, 2.0))
    else:
        mutation = tuple(float(np.clip(v, 0.0, 2.0)) for v in mutation)

    bounds = [(-1.0, 1.0)] * n_params

    best_state = None
    best_theta = None
    best_field = None
    best_perm = None
    best_sensitivity = None
    best_score = np.inf
    best_history: list[float] = []
    current_history: list[float] = []

    def evaluate_theta(theta: np.ndarray) -> float:
        nonlocal best_state, best_theta, best_field, best_perm, best_sensitivity, best_score

        state, field = parameterized_state_from_theta(theta, field_geometry, field_shape_cfg)
        perm, sensitivity, score = evaluate_state(
            fwd=fwd,
            state=state,
            variable_mask=variable_mask,
            low_cond=low_cond,
            high_cond=high_cond,
            adjacency=adjacency,
            electrode_supports=electrode_supports,
            enforce_connectivity=enforce_connectivity,
            repair_disconnected=repair_disconnected,
            benchmark_fn=benchmark_fn,
            benchmark_name=benchmark_name,
            benchmark_weights=benchmark_weights,
            mesh_centroids=mesh_centroids,
        )

        if best_state is None or score <= best_score:
            best_score = float(score)
            best_theta = np.asarray(theta, dtype=float).copy()
            best_state = np.asarray(state, dtype=bool).copy()
            best_field = np.asarray(field, dtype=float).copy()
            best_perm = np.asarray(perm, dtype=float).copy()
            best_sensitivity = np.asarray(sensitivity, dtype=float).copy()

        return float(score)

    generation_counter = {"value": 0}

    def de_callback(xk: np.ndarray, convergence: float) -> bool:
        _ = convergence
        generation_counter["value"] += 1
        current_score = evaluate_theta(xk)
        current_history.append(float(current_score))
        best_history.append(float(best_score))

        if progress_callback is not None:
            progress_callback(
                {
                    "generation": generation_counter["value"],
                    "n_generations": maxiter,
                    "best_score": float(best_score),
                    "current_score": float(current_score),
                    "best_theta": None if best_theta is None else np.asarray(best_theta, dtype=float).copy(),
                    "best_state": None if best_state is None else np.asarray(best_state, dtype=bool).copy(),
                    "best_field": None if best_field is None else np.asarray(best_field, dtype=float).copy(),
                    "best_perm": None if best_perm is None else np.asarray(best_perm, dtype=float).copy(),
                    "best_sensitivity": None if best_sensitivity is None else np.asarray(best_sensitivity, dtype=float).copy(),
                    "current_theta": np.asarray(xk, dtype=float).copy(),
                    "n_params": int(n_params),
                    "n_variable": int(np.count_nonzero(variable_mask)),
                }
            )

        return False

    result = differential_evolution(
        func=evaluate_theta,
        bounds=bounds,
        maxiter=maxiter,
        popsize=popsize,
        mutation=mutation,
        recombination=recombination,
        seed=int(seed),
        callback=de_callback,
        polish=False,
        disp=False,
        tol=0.0,
        atol=0.0,
        updating="deferred",
    )

    if best_theta is None:
        _ = evaluate_theta(result.x)
        current_history.append(float(best_score))
        best_history.append(float(best_score))

    return {
        "best_theta": best_theta,
        "best_state": best_state,
        "best_field": best_field,
        "best_perm": best_perm,
        "best_sensitivity": best_sensitivity,
        "best_score": float(best_score),
        "best_history": np.asarray(best_history, dtype=float),
        "current_history": np.asarray(current_history, dtype=float),
        "n_variable": int(np.count_nonzero(variable_mask)),
        "n_params": int(n_params),
    }


def _build_run_name(args: Any) -> str:
    custom = str(getattr(args, "run_name", "") or "").strip()
    if custom:
        return custom
    return f"{getattr(args, 'optimizer', 'de')}_{getattr(args, 'benchmark_name', 'uniformity')}_{time.strftime('%Y%m%d_%H%M%S')}"


def save_run_artifacts(
    output_dir: str,
    run_name: str,
    mesh_obj: Any,
    p1: list[float],
    p2: list[float],
    result: dict[str, Any],
    args: Any,
    field_shape_cfg: dict[str, Any],
    benchmark_weights: dict[str, float],
    connectivity_info: dict[str, Any],
) -> dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    base_path = os.path.join(output_dir, run_name)
    npz_path = f"{base_path}.npz"
    json_path = f"{base_path}.json"
    history_csv_path = f"{base_path}_history.csv"
    elements_csv_path = f"{base_path}_elements.csv"

    best_theta = result.get("best_theta")
    if best_theta is None:
        best_theta = np.empty((0,), dtype=float)

    np.savez_compressed(
        npz_path,
        node=np.asarray(mesh_obj.node, dtype=float),
        element=np.asarray(mesh_obj.element, dtype=int),
        el_pos=np.asarray(mesh_obj.el_pos, dtype=int),
        p1=np.asarray(p1, dtype=float),
        p2=np.asarray(p2, dtype=float),
        best_perm=np.asarray(result["best_perm"], dtype=float),
        best_sensitivity=np.asarray(result["best_sensitivity"], dtype=float),
        best_state=np.asarray(result["best_state"], dtype=bool),
        best_theta=np.asarray(best_theta, dtype=float),
        best_history=np.asarray(result["best_history"], dtype=float),
        current_history=np.asarray(result["current_history"], dtype=float),
    )

    with open(history_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "best_score", "current_score"])
        for idx, (best_val, curr_val) in enumerate(zip(np.asarray(result["best_history"]).ravel(), np.asarray(result["current_history"]).ravel()), start=1):
            writer.writerow([idx, float(best_val), float(curr_val)])

    with open(elements_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["element_index", "conductivity", "sensitivity", "is_high"])
        perm = np.asarray(result["best_perm"], dtype=float).ravel()
        sens = np.asarray(result["best_sensitivity"], dtype=float).ravel()
        high = np.asarray(result["best_state"], dtype=bool).ravel()
        for idx in range(perm.size):
            writer.writerow([idx, float(perm[idx]), float(sens[idx]), int(high[idx])])

    metadata = {
        "run_name": run_name,
        "optimizer": str(args.optimizer),
        "benchmark_name": str(args.benchmark_name),
        "benchmark_profile": str(args.benchmark_profile),
        "benchmark_weights": {k: float(v) for k, v in benchmark_weights.items() if isinstance(v, (int, float, np.floating, np.integer))},
        "seed": int(args.seed),
        "h0": float(args.h0),
        "low_cond": float(args.low_cond),
        "high_cond": float(args.high_cond),
        "n_elements": int(np.asarray(mesh_obj.element).shape[0]),
        "n_electrodes": int(np.asarray(mesh_obj.el_pos).size),
        "best_score": float(result["best_score"]),
        "best_sensitivity_min": float(np.min(np.asarray(result["best_sensitivity"], dtype=float))),
        "best_sensitivity_max": float(np.max(np.asarray(result["best_sensitivity"], dtype=float))),
        "p1": [float(v) for v in p1],
        "p2": [float(v) for v in p2],
        "connectivity": {
            "high_elements_connected": bool(connectivity_info["is_connected"]),
            "electrodes_connected_by_high_region": bool(connectivity_info["electrodes_linked"]),
            "high_connected_components": int(connectivity_info["components"]),
            "isolated_high_elements": int(connectivity_info["isolated_count"]),
        },
        "field_shape_cfg": {
            key: (
                [float(x) for x in np.asarray(val).ravel()]
                if isinstance(val, (list, tuple, np.ndarray))
                else bool(val)
                if isinstance(val, (bool, np.bool_))
                else int(val)
                if isinstance(val, (int, np.integer))
                else float(val)
                if isinstance(val, (float, np.floating))
                else str(val)
            )
            for key, val in field_shape_cfg.items()
            if key not in {"branching_electrode_points_norm", "branching_electrode_points_norm_right"}
        },
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return {
        "npz": npz_path,
        "metadata_json": json_path,
        "history_csv": history_csv_path,
        "elements_csv": elements_csv_path,
    }


def _build_element_adjacency_from_triangles(triangles: np.ndarray) -> list[list[int]]:
    tri = np.asarray(triangles, dtype=int)
    n_elem = int(tri.shape[0])
    adjacency = [[] for _ in range(n_elem)]
    for i in range(n_elem):
        vertices_i = set(tri[i].tolist())
        for j in range(i + 1, n_elem):
            vertices_j = set(tri[j].tolist())
            if len(vertices_i & vertices_j) >= 2:
                adjacency[i].append(j)
                adjacency[j].append(i)
    return adjacency


def load_saved_result(npz_path: str) -> dict[str, Any]:
    with np.load(npz_path, allow_pickle=False) as data:
        required = {"best_perm", "best_sensitivity", "best_state"}
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"Saved result is missing required arrays: {', '.join(missing)}")

        return {
            "best_perm": np.asarray(data["best_perm"], dtype=float).ravel(),
            "best_sensitivity": np.asarray(data["best_sensitivity"], dtype=float).ravel(),
            "best_state": np.asarray(data["best_state"], dtype=bool).ravel(),
            "node": np.asarray(data["node"], dtype=float) if "node" in data.files else None,
            "element": np.asarray(data["element"], dtype=int) if "element" in data.files else None,
            "el_pos": np.asarray(data["el_pos"], dtype=int).ravel() if "el_pos" in data.files else None,
        }


def benchmark_saved_result(npz_path: str, benchmark_name: str = "uniformity", benchmark_weights: dict[str, float] | None = None) -> dict[str, Any]:
    payload = load_saved_result(npz_path)
    sensitivity = payload["best_sensitivity"]
    state = payload["best_state"]
    adjacency = None
    mesh_centroids = None
    jacobian = None

    if payload.get("node") is not None and payload.get("element") is not None:
        node = np.asarray(payload["node"], dtype=float)
        element = np.asarray(payload["element"], dtype=int)
        mesh_centroids = np.mean(node[element], axis=1)[:, :2]
        adjacency = _build_element_adjacency_from_triangles(element)

    touch_metrics = {
        "expected_sensitivity",
        "minimax_sensitivity",
        "softmin_sensitivity",
        "snr_sensitivity",
        "distinguishability",
        "combined",
    }

    if benchmark_name in touch_metrics:
        if payload.get("node") is None or payload.get("element") is None or payload.get("el_pos") is None:
            raise ValueError("Touch benchmarks require node, element, and el_pos in NPZ.")

        mesh_obj = PyEITMesh(
            node=np.asarray(payload["node"], dtype=float),
            element=np.asarray(payload["element"], dtype=int),
            el_pos=np.asarray(payload["el_pos"], dtype=int),
            perm=np.asarray(payload["best_perm"], dtype=float),
        )
        n_el = int(np.asarray(mesh_obj.el_pos).size)
        protocol_obj = protocol.create(n_el=n_el, dist_exc=max(1, n_el // 2), step_meas=1, parser_meas="rotate_meas")
        fwd = EITForward(mesh_obj, protocol_obj)
        jacobian, _ = fwd.compute_jac(perm=np.asarray(payload["best_perm"], dtype=float))

    score = float(
        get_benchmark(benchmark_name)(
            sensitivity=sensitivity,
            state=state,
            adjacency=adjacency,
            weights=benchmark_weights,
            jacobian=jacobian,
            mesh_centroids=mesh_centroids,
        )
    )

    isolated_count = count_isolated_high_elements(state, adjacency) if adjacency is not None else None
    connected = has_connected_high_elements(state, adjacency) if adjacency is not None else None

    return {
        "source_npz": os.path.abspath(npz_path),
        "benchmark_name": str(benchmark_name),
        "n_elements": int(state.size),
        "n_high_elements": int(np.count_nonzero(state)),
        "score": float(score),
        "sensitivity_min": float(np.min(sensitivity)),
        "sensitivity_max": float(np.max(sensitivity)),
        "sensitivity_mean": float(np.mean(sensitivity)),
        "isolated_high_elements": None if isolated_count is None else int(isolated_count),
        "high_elements_connected": None if connected is None else bool(connected),
    }


def plot_results(
    mesh_obj: Any,
    p1: list[float],
    p2: list[float],
    best_perm: np.ndarray,
    best_sensitivity: np.ndarray,
    best_hist: np.ndarray,
    curr_hist: np.ndarray,
    best_score: float,
    best_theta: np.ndarray | None = None,
    field_shape_cfg: dict[str, Any] | None = None,
    comparison_rows: list[dict[str, Any]] | None = None,
) -> None:
    del best_theta, field_shape_cfg
    pts = np.asarray(mesh_obj.node, dtype=float)
    tri = np.asarray(mesh_obj.element, dtype=int)
    el_pos = np.asarray(mesh_obj.el_pos, dtype=int)

    rows = [
        {
            "name": "Best Optimized",
            "perm": np.asarray(best_perm).ravel(),
            "sensitivity": np.asarray(best_sensitivity).ravel(),
            "score": float(best_score),
        }
    ] + (comparison_rows or [])

    fig, axes = plt.subplots(len(rows), 3, figsize=(18, 5.0 * len(rows)))
    if len(rows) == 1:
        axes = np.asarray([axes])

    for r, row in enumerate(rows):
        ax0, ax1, ax2 = axes[r, 0], axes[r, 1], axes[r, 2]

        im0 = ax0.tripcolor(pts[:, 0], pts[:, 1], tri, row["perm"], shading="flat", cmap="RdBu_r")
        ax0.plot(pts[el_pos, 0], pts[el_pos, 1], "ko", markersize=3)
        ax0.set_aspect("equal")
        ax0.set_xlim(p1[0] - 0.2, p2[0] + 0.2)
        ax0.set_ylim(p1[1] - 0.2, p2[1] + 0.2)
        ax0.set_title(f"{row['name']} Conductivity")
        fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04, label="Conductivity")

        sens_display = sensitivity_for_display(np.asarray(row["sensitivity"], dtype=float))
        im1 = ax1.tripcolor(pts[:, 0], pts[:, 1], tri, sens_display, shading="flat", cmap="viridis")
        ax1.set_aspect("equal")
        ax1.set_xlim(p1[0] - 0.2, p2[0] + 0.2)
        ax1.set_ylim(p1[1] - 0.2, p2[1] + 0.2)
        ax1.set_title(f"{row['name']} Sensitivity (log10)")
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label="log10(sum |J[:, elem]|)")

        if r == 0:
            ax2.plot(np.asarray(best_hist), label="Best-so-far score", linewidth=1.8)
            ax2.plot(np.asarray(curr_hist), label="Current score", linewidth=1.0, alpha=0.7)
            ax2.set_title("Optimization History")
            ax2.set_xlabel("Generation")
            ax2.set_ylabel("Score")
            ax2.grid(alpha=0.25)
            ax2.legend(loc="best")
        else:
            ax2.axis("off")

    plt.tight_layout()
    plt.show()


@dataclass
class OptimizationContext:
    mesh_obj: Any
    protocol_obj: Any
    variable_mask: np.ndarray
    field_shape_cfg: dict[str, Any]
    p1: list[float]
    p2: list[float]
    benchmark_name: str
    benchmark_weights: dict[str, float]
    adjacency: list[list[int]]
    electrode_supports: list[np.ndarray]
    low_cond: float
    high_cond: float
    seed: int
    h0: float
    output_dir: str
    run_name: str


def _timestamp_run_name(prefix: str = "ui") -> str:
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}"


class CombinedOptimizerUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("BendingSim Optimizer")
        self.root.geometry("1600x920")

        self.mesh_obj = None
        self.protocol_obj = None
        self.variable_mask = None
        self.optimization_result: dict[str, Any] | None = None
        self.context: OptimizationContext | None = None
        self.optimization_thread: threading.Thread | None = None
        self.optimization_queue: queue.Queue = queue.Queue()
        self.is_running = False
        self.latest_progress: dict[str, Any] | None = None
        self.last_saved_npz_path: str | None = None
        self.live_best_history: list[float] = []
        self.live_current_history: list[float] = []
        self.reference_sensitivity: np.ndarray | None = None
        self.mesh_colorbar = None
        self.sensitivity_colorbar = None
        self.biomimetic_thread: threading.Thread | None = None

        self._build_variables()
        self._build_ui()
        self._poll_queue()

    def _build_variables(self) -> None:
        # Mesh setup
        self.n_el_var = tk.StringVar(value="16")
        self.h0_var = tk.StringVar(value="0.10")
        self.p1x_var = tk.StringVar(value="0.0")
        self.p1y_var = tk.StringVar(value="0.0")
        self.p2x_var = tk.StringVar(value="5.0")
        self.p2y_var = tk.StringVar(value="2.0")

        # Objective: sensitivity scoring parameters
        self.metric_option_var = tk.StringVar(value="Uniformity")
        self.sens_mean_target_var = tk.StringVar(value="auto")
        self.sens_variance_weight_var = tk.StringVar(value="1.0")
        self.enforce_connectivity_var = tk.BooleanVar(value=True)

        # Element model parameters
        self.model_option_var = tk.StringVar(value="Element Model")
        self.low_cond_var = tk.StringVar(value="1000")
        self.high_cond_var = tk.StringVar(value="500")
        self.seed_var = tk.StringVar(value="7")
        self.branching_depth_max_var = tk.StringVar(value="5")
        self.branching_max_children_var = tk.StringVar(value="2")
        self.branching_angle_frac_var = tk.StringVar(value="0.34")
        self.branching_child_angle_frac_var = tk.StringVar(value="0.34")
        self.branching_size_min_frac_var = tk.StringVar(value="0.55")
        self.branching_size_max_frac_var = tk.StringVar(value="1.0")
        self.branching_seed_all_electrodes_var = tk.BooleanVar(value=True)

        # Optimization strategy parameters (DE only)
        self.strategy_option_var = tk.StringVar(value="DE")
        self.repair_disconnected_var = tk.BooleanVar(value=False)
        self.de_maxiter_var = tk.StringVar(value="20")
        self.de_popsize_var = tk.StringVar(value="8")
        self.de_mutation_min_var = tk.StringVar(value="0.5")
        self.de_mutation_max_var = tk.StringVar(value="1.0")
        self.de_recombination_var = tk.StringVar(value="0.7")
        self.de_seed_var = tk.StringVar(value="42")

        # Biomimetic analysis parameters
        self.fungal_sigma_0_var = tk.StringVar(value="1.0")
        self.fungal_alpha_var = tk.StringVar(value="1.5")
        self.fungal_rho_var = tk.StringVar(value="0.99")
        self.fungal_agents_var = tk.StringVar(value="10")
        self.fungal_steps_var = tk.StringVar(value="100")
        self.fungal_fem_every_var = tk.StringVar(value="10")
        self.fungal_sigma_max_var = tk.StringVar(value="")
        self.fungal_normalise_var = tk.BooleanVar(value=False)

        self.touch_radius_var = tk.StringVar(value="0.30")
        self.touch_samples_var = tk.StringVar(value="200")
        self.touch_temperature_var = tk.StringVar(value="1.0")
        self.touch_lambda_var = tk.StringVar(value="0.1")
        self.touch_noise_var = tk.StringVar(value="")

        # Run/output settings
        self.save_results_var = tk.BooleanVar(value=True)
        self.output_dir_var = tk.StringVar(value=str(Path("results").resolve()))
        self.run_name_var = tk.StringVar(value=timestamp_run_name())

        # Status variables
        self.status_var = tk.StringVar(value="Generate a mesh to begin.")
        self.progress_var = tk.StringVar(value="Idle")
        self.best_score_var = tk.StringVar(value="Best score: n/a")
        self.current_score_var = tk.StringVar(value="Current score: n/a")
        self.generation_var = tk.StringVar(value="Generation: n/a")
        self.param_count_var = tk.StringVar(value="Parameters: n/a")

        # Benchmark variables
        self.benchmark_npz_var = tk.StringVar(value="")
        self.benchmark_score_json_var = tk.StringVar(value="")
        self.benchmark_result_var = tk.StringVar(value="No benchmark run yet.")

    def _build_ui(self) -> None:
        outer = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        outer.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(outer, padding=8)
        right = ttk.Frame(outer, padding=8)
        outer.add(left, weight=1)
        outer.add(right, weight=3)

        controls = ttk.Notebook(left)
        controls.pack(fill=tk.BOTH, expand=True)

        mesh_tab, mesh_canvas, mesh_inner = self._create_scrollable_tab(controls)
        optimize_tab, optimize_canvas, optimize_inner = self._create_scrollable_tab(controls)
        save_tab, save_canvas, save_inner = self._create_scrollable_tab(controls)
        benchmark_tab, benchmark_canvas, benchmark_inner = self._create_scrollable_tab(controls)
        biomimetic_tab, biomimetic_canvas, biomimetic_inner = self._create_scrollable_tab(controls)
        controls.add(mesh_tab, text="Mesh")
        controls.add(optimize_tab, text="Optimization")
        controls.add(save_tab, text="Save")
        controls.add(benchmark_tab, text="Benchmark")
        controls.add(biomimetic_tab, text="Biomimetic")

        self._build_mesh_tab(mesh_inner)
        self._build_optimize_tab(optimize_inner)
        self._build_save_tab(save_inner)
        self._build_benchmark_tab(benchmark_inner)
        self._build_biomimetic_tab(biomimetic_inner)
        self._enable_scroll_wheel(mesh_inner, mesh_canvas)
        self._enable_scroll_wheel(optimize_inner, optimize_canvas)
        self._enable_scroll_wheel(save_inner, save_canvas)
        self._enable_scroll_wheel(benchmark_inner, benchmark_canvas)
        self._enable_scroll_wheel(biomimetic_inner, biomimetic_canvas)

        status_frame = ttk.LabelFrame(left, text="Live Status", padding=8)
        status_frame.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(status_frame, textvariable=self.status_var, wraplength=370).pack(anchor="w")
        ttk.Label(status_frame, textvariable=self.progress_var).pack(anchor="w", pady=(4, 0))
        ttk.Label(status_frame, textvariable=self.generation_var).pack(anchor="w")
        ttk.Label(status_frame, textvariable=self.best_score_var).pack(anchor="w")
        ttk.Label(status_frame, textvariable=self.current_score_var).pack(anchor="w")
        ttk.Label(status_frame, textvariable=self.param_count_var).pack(anchor="w")

        self.figure = Figure(figsize=(11.2, 8.6), dpi=100)
        gs = self.figure.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.3, wspace=0.3)
        self.ax_mesh = self.figure.add_subplot(gs[0, 0])
        self.ax_sensitivity = self.figure.add_subplot(gs[0, 1])
        self.ax_history = self.figure.add_subplot(gs[1, :])
        self.canvas = FigureCanvasTkAgg(self.figure, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._draw_idle_state()

    def _build_mesh_tab(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Mesh Generation", padding=8)
        frame.pack(fill=tk.X, pady=(0, 8))

        self._row_entry(frame, 0, "Electrodes", self.n_el_var)
        self._row_entry(frame, 1, "Mesh h0", self.h0_var)
        self._row_entry(frame, 2, "p1 x", self.p1x_var)
        self._row_entry(frame, 3, "p1 y", self.p1y_var)
        self._row_entry(frame, 4, "p2 x", self.p2x_var)
        self._row_entry(frame, 5, "p2 y", self.p2y_var)

        ttk.Button(frame, text="Generate Mesh", command=self.generate_mesh).grid(row=6, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Button(frame, text="Reset Run Name", command=lambda: self.run_name_var.set(timestamp_run_name())).grid(row=6, column=2, columnspan=2, sticky="ew", pady=(8, 0))

        self.mesh_info_var = tk.StringVar(value="No mesh generated yet.")
        ttk.Label(parent, textvariable=self.mesh_info_var, wraplength=380).pack(anchor="w")

    def _build_optimize_tab(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Optimization", padding=8)
        frame.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(
            frame,
            text="Select the metric, model, and strategy. Each section exposes the parameters for the selected option.",
            wraplength=380,
        ).pack(anchor="w", pady=(0, 8))

        self._build_option_section(
            frame,
            title="Metric",
            selector_label="Metric",
            selector_var=self.metric_option_var,
            selector_values=(
                "Uniformity",
                "Expected Sensitivity",
                "Minimax Sensitivity",
                "Softmin Sensitivity",
                "SNR Sensitivity",
                "Distinguishability",
                "Combined",
            ),
            selector_hint="Pick the optimisation score. Touch-sensitivity scores use the biomimetic Jacobian-based metrics.",
            builder=self._build_metric_options,
        )
        self._build_option_section(
            frame,
            title="Model",
            selector_label="Model",
            selector_var=self.model_option_var,
            selector_values=("Element Model", "Fungal Growth", "Branching Model"),
            selector_hint="Choose element-wise conductivity or a growth model seeded from the mesh boundary.",
            builder=self._build_model_options,
        )
        self._build_option_section(
            frame,
            title="Strategy",
            selector_label="Strategy",
            selector_var=self.strategy_option_var,
            selector_values=("DE",),
            selector_hint="Differential evolution is the only supported strategy in this UI.",
            builder=self._build_strategy_options,
        )

        button_row = ttk.Frame(parent)
        button_row.pack(fill=tk.X, pady=(8, 0))
        self.optimize_button = ttk.Button(button_row, text="Run Optimization", command=self.start_optimization)
        self.optimize_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 4))
        self.stop_button = ttk.Button(button_row, text="Stop After Current Generation", command=self.request_stop, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(4, 0))

    def _build_option_section(
        self,
        parent: ttk.Frame,
        title: str,
        selector_label: str,
        selector_var: tk.StringVar,
        selector_values: tuple[str, ...],
        selector_hint: str,
        builder,
    ) -> None:
        section = ttk.LabelFrame(parent, text=title, padding=8)
        section.pack(fill=tk.X, pady=(0, 8))

        selector_row = ttk.Frame(section)
        selector_row.pack(fill=tk.X)
        selector_row.columnconfigure(1, weight=1)

        ttk.Label(selector_row, text=selector_label).grid(row=0, column=0, sticky="w")
        ttk.Combobox(selector_row, textvariable=selector_var, values=list(selector_values), state="readonly").grid(
            row=0,
            column=1,
            sticky="ew",
            padx=(8, 0),
        )

        ttk.Label(section, text=selector_hint, wraplength=380, font=("TkDefaultFont", 8)).pack(anchor="w", pady=(4, 0))

        details = ttk.Frame(section)
        details.pack(fill=tk.X, pady=(8, 0))

        def refresh(*_args: Any) -> None:
            for child in details.winfo_children():
                child.destroy()
            builder(details, selector_var.get())

        selector_var.trace_add("write", refresh)
        refresh()

    def _build_metric_options(self, parent: ttk.Frame, selected: str) -> None:
        if selected == "Uniformity":
            ttk.Label(
                parent,
                text="Uniformity uses the Jacobian sensitivity field and minimizes its variance.",
                wraplength=360,
            ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))
            self._row_entry(parent, 1, "Sensitivity mean target", self.sens_mean_target_var)
            ttk.Label(parent, text="(set to 'auto' for automatic)", font=("TkDefaultFont", 8)).grid(row=2, column=0, columnspan=2, sticky="w", pady=(0, 8))
            self._row_entry(parent, 3, "Uniformity weight", self.sens_variance_weight_var)
            ttk.Checkbutton(parent, text="Enforce connectivity (reject invalid)", variable=self.enforce_connectivity_var).grid(row=4, column=0, columnspan=2, sticky="w", pady=(6, 0))
            return

        metric_descriptions = {
            "Expected Sensitivity": "Maximize the mean touch sensitivity over sampled patches.",
            "Minimax Sensitivity": "Maximize the worst-case touch sensitivity over sampled patches.",
            "Softmin Sensitivity": "Use a smooth minimum over patch energies.",
            "SNR Sensitivity": "Maximize the patch-wise signal-to-noise score.",
            "Distinguishability": "Maximize patch-to-patch signal separation.",
            "Combined": "Balance mean sensitivity with a spatial uniformity penalty.",
        }
        ttk.Label(
            parent,
            text=metric_descriptions.get(selected, f"Unsupported metric: {selected}"),
            wraplength=360,
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))
        ttk.Label(parent, text="Touch-sensitivity metrics use the biomimetic analysis controls below.", wraplength=360, font=("TkDefaultFont", 8)).grid(row=1, column=0, columnspan=2, sticky="w")

    def _build_model_options(self, parent: ttk.Frame, selected: str) -> None:
        if selected == "Element Model":
            ttk.Label(
                parent,
                text="The element model assigns one parameter per variable element.",
                wraplength=360,
            ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))
            self._row_entry(parent, 1, "Low cond", self.low_cond_var)
            self._row_entry(parent, 2, "High cond", self.high_cond_var)
            self._row_entry(parent, 3, "Seed", self.seed_var)
            return

        if selected not in {"Fungal Growth", "Branching Model"}:
            ttk.Label(parent, text=f"Unsupported model: {selected}").pack(anchor="w")
            return

        if selected == "Fungal Growth":
            description = "The fungal growth model grows a branching conductivity pattern from the electrode region."
        else:
            description = "The branching model grows high-conductivity paths from the electrode region."

        ttk.Label(
            parent,
            text=description,
            wraplength=360,
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))
        self._row_entry(parent, 1, "Low cond", self.low_cond_var)
        self._row_entry(parent, 2, "High cond", self.high_cond_var)
        self._row_entry(parent, 3, "Seed", self.seed_var)
        ttk.Separator(parent, orient=tk.HORIZONTAL).grid(row=4, column=0, columnspan=2, sticky="ew", pady=(8, 8))
        ttk.Label(parent, text="Branching controls", font=("TkDefaultFont", 9, "bold")).grid(row=5, column=0, columnspan=2, sticky="w", pady=(0, 4))
        self._row_entry(parent, 6, "Depth max", self.branching_depth_max_var)
        self._row_entry(parent, 7, "Max children", self.branching_max_children_var)
        self._row_entry(parent, 8, "Angle frac", self.branching_angle_frac_var)
        self._row_entry(parent, 9, "Child angle frac", self.branching_child_angle_frac_var)
        self._row_entry(parent, 10, "Size min frac", self.branching_size_min_frac_var)
        self._row_entry(parent, 11, "Size max frac", self.branching_size_max_frac_var)
        ttk.Checkbutton(parent, text="Seed from all electrodes", variable=self.branching_seed_all_electrodes_var).grid(row=12, column=0, columnspan=2, sticky="w", pady=(6, 0))

    def _build_strategy_options(self, parent: ttk.Frame, selected: str) -> None:
        if selected != "DE":
            ttk.Label(parent, text=f"Unsupported strategy: {selected}").pack(anchor="w")
            return

        ttk.Label(
            parent,
            text="Differential evolution is used to search the element-wise conductivity states.",
            wraplength=360,
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))
        ttk.Checkbutton(parent, text="Repair disconnected (constrain search space)", variable=self.repair_disconnected_var).grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 8))
        ttk.Label(parent, text="DE Parameters", font=("TkDefaultFont", 9, "bold")).grid(row=2, column=0, columnspan=2, sticky="w", pady=(0, 4))
        self._row_entry(parent, 3, "Max iterations", self.de_maxiter_var)
        self._row_entry(parent, 4, "Population size", self.de_popsize_var)
        self._row_entry(parent, 5, "Mutation min", self.de_mutation_min_var)
        self._row_entry(parent, 6, "Mutation max", self.de_mutation_max_var)
        self._row_entry(parent, 7, "Recombination", self.de_recombination_var)
        self._row_entry(parent, 8, "DE seed", self.de_seed_var)

    def _build_save_tab(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Artifact Saving", padding=8)
        frame.pack(fill=tk.X, pady=(0, 8))
        self._row_entry(frame, 0, "Output dir", self.output_dir_var)
        self._row_entry(frame, 1, "Run name", self.run_name_var)
        ttk.Checkbutton(frame, text="Save results after optimization", variable=self.save_results_var).grid(row=2, column=0, columnspan=2, sticky="w", pady=(6, 0))

        ttk.Button(frame, text="Browse output dir", command=self.browse_output_dir).grid(row=3, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Button(frame, text="Save Current Results Now", command=self.save_current_results).grid(row=4, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        ttk.Button(frame, text="Open Detailed Result Plot", command=self.open_detailed_result_plot).grid(row=5, column=0, columnspan=2, sticky="ew", pady=(4, 0))

        self.save_info_var = tk.StringVar(value="No saved optimization run yet.")
        ttk.Label(parent, textvariable=self.save_info_var, wraplength=380).pack(anchor="w")

    def _build_benchmark_tab(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Post-save Benchmarking", padding=8)
        frame.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(
            frame,
            text="Load a saved NPZ and score the exported conductivity state with the selected benchmark.",
            wraplength=360,
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))

        self._row_entry(frame, 1, "Saved NPZ", self.benchmark_npz_var)
        self._row_entry(frame, 2, "Score JSON", self.benchmark_score_json_var)

        ttk.Button(frame, text="Browse NPZ", command=self.browse_benchmark_npz).grid(row=3, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Button(frame, text="Score Saved Result", command=self.run_saved_benchmark).grid(row=4, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        ttk.Button(frame, text="Use Last Saved Run", command=self.load_last_saved_run_for_benchmark).grid(row=5, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        ttk.Button(frame, text="Use Current Saved Run", command=self.use_current_saved_run_for_benchmark).grid(row=6, column=0, columnspan=2, sticky="ew", pady=(4, 0))

        ttk.Label(parent, textvariable=self.benchmark_result_var, wraplength=380).pack(anchor="w")

    def _build_biomimetic_tab(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Biomimetic Analysis", padding=8)
        frame.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(
            frame,
            text="Touch sensitivity uses the current Jacobian as input. Fungal growth parameters are configured in the Optimization tab.",
            wraplength=360,
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))

        ttk.Label(frame, text="Touch Sensitivity", font=("TkDefaultFont", 9, "bold")).grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 4))
        self._row_entry(frame, 2, "Touch radius", self.touch_radius_var)
        self._row_entry(frame, 3, "Patch samples", self.touch_samples_var)
        self._row_entry(frame, 4, "Softmin temperature", self.touch_temperature_var)
        self._row_entry(frame, 5, "Uniformity lambda", self.touch_lambda_var)
        self._row_entry(frame, 6, "Noise scale", self.touch_noise_var)

        button_row = ttk.Frame(parent)
        button_row.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(button_row, text="Run Biomimetic Analysis", command=self.start_biomimetic_analysis).pack(fill=tk.X)

        self.biomimetic_status_var = tk.StringVar(value="Generate a mesh before running biomimetic analysis.")
        self.biomimetic_result_var = tk.StringVar(value="No biomimetic analysis run yet.")
        ttk.Label(parent, textvariable=self.biomimetic_status_var, wraplength=380).pack(anchor="w")
        ttk.Label(parent, textvariable=self.biomimetic_result_var, wraplength=380).pack(anchor="w", pady=(4, 0))

    def _mesh_centroids(self) -> np.ndarray:
        if self.mesh_obj is None:
            raise ValueError("Generate a mesh first.")
        pts = np.asarray(self.mesh_obj.node, dtype=float)
        tri = np.asarray(self.mesh_obj.element, dtype=int)
        return np.mean(pts[tri], axis=1)[:, :2]

    def _mesh_areas(self) -> np.ndarray:
        if self.mesh_obj is None:
            raise ValueError("Generate a mesh first.")
        pts = np.asarray(self.mesh_obj.node, dtype=float)
        tri = np.asarray(self.mesh_obj.element, dtype=int)
        p0 = pts[tri[:, 0], :2]
        p1 = pts[tri[:, 1], :2]
        p2 = pts[tri[:, 2], :2]
        return 0.5 * np.abs(np.cross(p1 - p0, p2 - p0))

    def _current_jacobian(self, perm: np.ndarray | None = None) -> np.ndarray:
        if self.mesh_obj is None or self.protocol_obj is None:
            raise ValueError("Generate a mesh first.")
        if perm is None:
            perm = np.ones(int(np.asarray(self.mesh_obj.element).shape[0]), dtype=float)
        fwd = EITForward(self.mesh_obj, self.protocol_obj)
        jacobian, _ = fwd.compute_jac(perm=np.asarray(perm, dtype=float))
        return np.asarray(jacobian, dtype=float)

    def _touch_noise_covariance(self, n_measurements: int) -> np.ndarray | None:
        value = str(self.touch_noise_var.get()).strip()
        if not value:
            return None
        scale = parse_float(self.touch_noise_var, "touch noise scale")
        if scale <= 0.0:
            return None
        return np.eye(n_measurements, dtype=float) * (scale**2)

    def _build_proxy_growth_mesh(self):
        if self.mesh_obj is None or self.protocol_obj is None:
            raise ValueError("Generate a mesh first.")

        adjacency = build_element_adjacency(self.mesh_obj)
        areas = self._mesh_areas()
        fwd = EITForward(self.mesh_obj, self.protocol_obj)
        mesh_obj = self.mesh_obj

        class ProxyGrowthMesh:
            def __init__(self) -> None:
                self.n_elements = int(np.asarray(mesh_obj.element).shape[0])
                self.areas = np.asarray(areas, dtype=float)

            def solve(self, sigma):
                jacobian, _ = fwd.compute_jac(perm=np.asarray(sigma, dtype=float))
                return np.asarray(jacobian, dtype=float)

            def current_density(self, phi):
                field = np.asarray(phi, dtype=float)
                return np.log10(np.sum(np.abs(field), axis=0) + np.finfo(float).tiny)

            def gradient_sq(self, phi):
                density = self.current_density(phi)
                return np.asarray(density, dtype=float) ** 2

            def adjacent_elements(self, elem_idx):
                return adjacency[int(elem_idx)]

        return ProxyGrowthMesh()

    def start_biomimetic_analysis(self) -> None:
        if self.mesh_obj is None or self.protocol_obj is None:
            messagebox.showwarning("Missing mesh", "Generate a mesh before running biomimetic analysis.")
            return

        if self.biomimetic_thread is not None and self.biomimetic_thread.is_alive():
            messagebox.showinfo("Analysis running", "Biomimetic analysis is already in progress.")
            return

        self.biomimetic_status_var.set("Running biomimetic analysis...")
        self.biomimetic_result_var.set("Working...")

        def worker() -> None:
            try:
                jacobian = self._current_jacobian()
                centroids = self._mesh_centroids()
                touch_radius = parse_float(self.touch_radius_var, "touch radius")
                touch_samples = parse_int(self.touch_samples_var, "touch samples")
                touch_temperature = parse_float(self.touch_temperature_var, "softmin temperature")
                touch_lambda = parse_float(self.touch_lambda_var, "uniformity lambda")
                noise_cov = self._touch_noise_covariance(jacobian.shape[0])

                touch_cost = TouchSensitivityCost(
                    J=jacobian,
                    mesh_elements=centroids,
                    touch_radius=touch_radius,
                    noise_cov=noise_cov,
                )

                touch_metrics = {
                    "expected": touch_cost.expected_sensitivity(touch_samples),
                    "minimax": touch_cost.minimax_sensitivity(touch_samples),
                    "softmin": touch_cost.softmin_sensitivity(touch_samples, temperature=touch_temperature),
                    "snr": touch_cost.snr_sensitivity(touch_samples),
                    "distinguishability": touch_cost.distinguishability(max(10, touch_samples // 2)),
                    "combined": touch_cost.combined(touch_samples, lambda_uniformity=touch_lambda),
                }

                growth_mesh = self._build_proxy_growth_mesh()
                sigma_max_value = str(self.fungal_sigma_max_var.get()).strip()
                fungal = FungalGrowthEIT(
                    mesh=growth_mesh,
                    sigma_0=parse_float(self.fungal_sigma_0_var, "sigma_0"),
                    alpha=parse_float(self.fungal_alpha_var, "alpha"),
                    rho=parse_float(self.fungal_rho_var, "rho"),
                    n_agents=parse_int(self.fungal_agents_var, "agents"),
                    n_steps=parse_int(self.fungal_steps_var, "steps"),
                    fem_solve_every=parse_int(self.fungal_fem_every_var, "fem solve every"),
                    sigma_max=None if not sigma_max_value else parse_float(self.fungal_sigma_max_var, "sigma max"),
                    normalise=bool(self.fungal_normalise_var.get()),
                )
                fungal_history = fungal.grow(3)
                fungal_sigma = fungal.conductivity()

                self.optimization_queue.put(
                    {
                        "type": "biomimetic_done",
                        "payload": {
                            "touch_metrics": touch_metrics,
                            "touch_sensitivity": np.asarray(np.sum(np.abs(jacobian), axis=0), dtype=float),
                            "fungal_history": np.asarray(fungal_history, dtype=float),
                            "fungal_sigma": np.asarray(fungal_sigma, dtype=float),
                        },
                    }
                )
            except Exception as exc:
                self.optimization_queue.put(
                    {
                        "type": "biomimetic_error",
                        "payload": f"{exc}\n\n{traceback.format_exc()}",
                    }
                )

        self.biomimetic_thread = threading.Thread(target=worker, daemon=True)
        self.biomimetic_thread.start()

    def _create_scrollable_tab(self, notebook: ttk.Notebook) -> tuple[ttk.Frame, tk.Canvas, ttk.Frame]:
        outer = ttk.Frame(notebook)
        canvas = tk.Canvas(outer, highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer, orient=tk.VERTICAL, command=canvas.yview)
        inner = ttk.Frame(canvas)

        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        window_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _on_inner_configure(_event: tk.Event) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_configure(event: tk.Event) -> None:
            canvas.itemconfigure(window_id, width=event.width)

        inner.bind("<Configure>", _on_inner_configure)
        canvas.bind("<Configure>", _on_canvas_configure)
        return outer, canvas, inner

    def _enable_scroll_wheel(self, widget: ttk.Widget | tk.Widget, canvas: tk.Canvas) -> None:
        def _on_mousewheel(event: tk.Event) -> str:
            delta = int(getattr(event, "delta", 0))
            if delta == 0:
                return "break"

            steps = int(-delta / 120)
            if steps == 0:
                steps = -1 if delta > 0 else 1

            canvas.yview_scroll(steps, "units")
            return "break"

        def _bind_recursively(current: tk.Widget) -> None:
            current.bind("<MouseWheel>", _on_mousewheel)
            for child in current.winfo_children():
                _bind_recursively(child)

        _bind_recursively(widget)

    def _row_entry(self, parent: ttk.Frame, row: int, label: str, variable: tk.Variable) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        ttk.Entry(parent, textvariable=variable, width=24).grid(row=row, column=1, sticky="ew", padx=(6, 0), pady=2)
        parent.columnconfigure(1, weight=1)

    def _set_status(self, text: str) -> None:
        self.status_var.set(text)

    def _draw_idle_state(self) -> None:
        self.ax_mesh.clear()
        self.ax_mesh.set_title("Mesh preview")
        self.ax_mesh.text(0.5, 0.5, "Generate a mesh to begin", ha="center", va="center", transform=self.ax_mesh.transAxes)
        self.ax_mesh.set_axis_off()

        self.ax_sensitivity.clear()
        self.ax_sensitivity.set_title("Sensitivity preview")
        self.ax_sensitivity.text(0.5, 0.5, "Run optimization to see live sensitivity", ha="center", va="center", transform=self.ax_sensitivity.transAxes)
        self.ax_sensitivity.set_axis_off()

        self.ax_history.clear()
        self.ax_history.set_title("Optimization history")
        self.ax_history.text(0.5, 0.5, "Run optimization to see live scores", ha="center", va="center", transform=self.ax_history.transAxes)
        self.ax_history.set_axis_off()
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _mesh_bounds(self) -> tuple[list[float], list[float]]:
        p1 = [parse_float(self.p1x_var, "p1 x"), parse_float(self.p1y_var, "p1 y")]
        p2 = [parse_float(self.p2x_var, "p2 x"), parse_float(self.p2y_var, "p2 y")]
        if p1[0] >= p2[0] or p1[1] >= p2[1]:
            raise ValueError("Mesh bounds must satisfy p1 < p2 on both axes.")
        return p1, p2

    def _selected_model_key(self) -> str:
        model = self.model_option_var.get().strip().lower()
        if model == "element model":
            return "element"
        if model == "fungal growth":
            return "fungal"
        if model == "branching model":
            return "branching"
        raise ValueError(f"Unsupported model selected: {self.model_option_var.get()}")

    def _selected_benchmark_name(self) -> str:
        metric = self.metric_option_var.get().strip().lower()
        mapping = {
            "uniformity": "uniformity",
            "expected sensitivity": "expected_sensitivity",
            "minimax sensitivity": "minimax_sensitivity",
            "softmin sensitivity": "softmin_sensitivity",
            "snr sensitivity": "snr_sensitivity",
            "distinguishability": "distinguishability",
            "combined": "combined",
        }
        if metric not in mapping:
            raise ValueError(f"Unsupported metric selected: {self.metric_option_var.get()}")
        return mapping[metric]

    def _build_benchmark_weights(self) -> dict[str, float]:
        weights: dict[str, float] = {
            "uniformity_weight": parse_float(self.sens_variance_weight_var, "uniformity weight"),
            "touch_radius": parse_float(self.touch_radius_var, "touch radius"),
            "touch_samples": float(parse_int(self.touch_samples_var, "touch samples")),
            "touch_temperature": parse_float(self.touch_temperature_var, "softmin temperature"),
            "touch_lambda": parse_float(self.touch_lambda_var, "uniformity lambda"),
        }

        noise_text = str(self.touch_noise_var.get()).strip()
        if noise_text:
            weights["touch_noise_scale"] = parse_float(self.touch_noise_var, "noise scale")

        return weights

    def _build_field_shape_cfg(self) -> dict[str, Any]:
        model = self._selected_model_key()
        cfg: dict[str, Any] = {"model": model}
        if model == "branching":
            cfg.update(
                {
                    "branching_depth_max": parse_int(self.branching_depth_max_var, "branching depth max"),
                    "branching_max_children": parse_int(self.branching_max_children_var, "branching max children"),
                    "branching_angle_frac": parse_float(self.branching_angle_frac_var, "branching angle frac"),
                    "branching_child_angle_frac": parse_float(self.branching_child_angle_frac_var, "branching child angle frac"),
                    "branching_size_min_frac": parse_float(self.branching_size_min_frac_var, "branching size min frac"),
                    "branching_size_max_frac": parse_float(self.branching_size_max_frac_var, "branching size max frac"),
                    "branching_seed_all_electrodes": bool(self.branching_seed_all_electrodes_var.get()),
                }
            )
        return cfg

    def _build_field_geometry(self, field_shape_cfg: dict[str, Any]) -> dict[str, Any]:
        if self.mesh_obj is None or self.variable_mask is None:
            raise ValueError("Generate a mesh before building field geometry.")
        p1, p2 = self._mesh_bounds()
        return build_field_geometry(
            mesh_obj=self.mesh_obj,
            variable_mask=self.variable_mask,
            p1=p1,
            p2=p2,
            field_shape_cfg=field_shape_cfg,
        )

    def _build_context(self) -> OptimizationContext:
        if self.mesh_obj is None or self.variable_mask is None or self.protocol_obj is None:
            raise ValueError("Generate a mesh before starting optimization.")

        metric = self.metric_option_var.get()
        model = self.model_option_var.get()
        strategy = self.strategy_option_var.get()
        if metric not in {"Uniformity", "Expected Sensitivity", "Minimax Sensitivity", "Softmin Sensitivity", "SNR Sensitivity", "Distinguishability", "Combined"}:
            raise ValueError(f"Unsupported metric selected: {metric}")
        if model not in {"Element Model", "Fungal Growth", "Branching Model"}:
            raise ValueError(f"Unsupported model selected: {model}")
        if strategy != "DE":
            raise ValueError(f"Unsupported strategy selected: {strategy}")

        p1, p2 = self._mesh_bounds()
        field_shape_cfg = self._build_field_shape_cfg()
        field_geometry = self._build_field_geometry(field_shape_cfg)
        n_params = int(parameter_vector_size_from_cfg(field_shape_cfg, field_geometry))

        adjacency = build_element_adjacency(self.mesh_obj)
        electrode_supports = build_electrode_support_elements(
            self.mesh_obj,
            self.variable_mask,
            k_nearest=4,  # fixed value since no UI control needed
        )

        return OptimizationContext(
            mesh_obj=self.mesh_obj,
            protocol_obj=self.protocol_obj,
            variable_mask=self.variable_mask,
            field_shape_cfg=field_shape_cfg,
            p1=p1,
            p2=p2,
            benchmark_name=self._selected_benchmark_name(),
            benchmark_weights=self._build_benchmark_weights(),
            adjacency=adjacency,
            electrode_supports=electrode_supports,
            low_cond=parse_float(self.low_cond_var, "low conductivity"),
            high_cond=parse_float(self.high_cond_var, "high conductivity"),
            seed=parse_int(self.seed_var, "seed"),
            h0=parse_float(self.h0_var, "mesh h0"),
            output_dir=str(Path(self.output_dir_var.get()).expanduser()),
            run_name=str(self.run_name_var.get()).strip(),
        )

    def generate_mesh(self) -> None:
        try:
            n_el = parse_int(self.n_el_var, "electrode count")
            h0 = parse_float(self.h0_var, "mesh h0")
            p1, p2 = self._mesh_bounds()
            if n_el < 2:
                raise ValueError("Electrode count must be at least 2.")
        except ValueError as exc:
            messagebox.showerror("Invalid mesh input", str(exc))
            return

        try:
            self.mesh_obj, self.protocol_obj, self.variable_mask, p1_built, p2_built = build_model(n_el=n_el, h0=h0, p1=p1, p2=p2)
            self.reference_sensitivity = compute_reference_sensitivity(self.mesh_obj, self.protocol_obj)
            self.context = None
            self.optimization_result = None
            self.latest_progress = None
            self.live_best_history = []
            self.live_current_history = []

            n_elements = int(np.asarray(self.mesh_obj.element).shape[0])
            n_measurements = int(np.asarray(self.protocol_obj.meas_mat).shape[0])
            n_variable = int(np.count_nonzero(self.variable_mask))
            field_shape_cfg = self._build_field_shape_cfg()
            field_geometry = self._build_field_geometry(field_shape_cfg)
            n_params = int(parameter_vector_size_from_cfg(field_shape_cfg, field_geometry))
            self.mesh_info_var.set(
                f"Mesh ready: {n_elements} elements, {len(self.mesh_obj.el_pos)} electrodes, {n_measurements} measurements, {n_variable} variable elements."
            )
            self.param_count_var.set(f"Parameters: {n_params} ({field_shape_cfg['model']})")
            self.status_var.set("Mesh generated. Configure optimization and run the solver.")
            self._render_mesh()
        except Exception as exc:
            messagebox.showerror("Mesh generation failed", str(exc))

    def start_optimization(self) -> None:
        if self.is_running:
            messagebox.showinfo("Optimization running", "Optimization is already in progress.")
            return

        try:
            context = self._build_context()
        except Exception as exc:
            messagebox.showerror("Invalid optimization settings", str(exc))
            return

        self.context = context
        self.optimization_result = None
        self.latest_progress = None
        self.live_best_history = []
        self.live_current_history = []
        self.is_running = True
        self.optimize_button.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.NORMAL)
        self.progress_var.set("Optimization started.")
        self.status_var.set(
            f"Running {self.strategy_option_var.get()} optimization for {self.metric_option_var.get().lower()}..."
        )
        self.best_score_var.set("Best score: starting")
        self.current_score_var.set("Current score: starting")

        def progress_callback(payload: dict[str, Any]) -> None:
            self.optimization_queue.put({"type": "progress", "payload": payload})

        de_maxiter = parse_int(self.de_maxiter_var, "DE maxiter")
        de_popsize = parse_int(self.de_popsize_var, "DE popsize")
        de_mutation = (
            parse_float(self.de_mutation_min_var, "DE mutation min"),
            parse_float(self.de_mutation_max_var, "DE mutation max"),
        )
        de_recombination = parse_float(self.de_recombination_var, "DE recombination")
        de_seed = parse_int(self.de_seed_var, "DE seed")
        enforce_connectivity = bool(self.enforce_connectivity_var.get())
        repair_disconnected = bool(self.repair_disconnected_var.get())

        def worker() -> None:
            try:
                field_geometry = self._build_field_geometry(context.field_shape_cfg)
                n_params = int(parameter_vector_size_from_cfg(context.field_shape_cfg, field_geometry))
                self.optimization_queue.put({"type": "meta", "payload": {"n_params": n_params}})

                fwd = EITForward(context.mesh_obj, context.protocol_obj)

                result = optimize_parameterized_field_de(
                    fwd=fwd,
                    variable_mask=context.variable_mask,
                    field_geometry=field_geometry,
                    field_shape_cfg=context.field_shape_cfg,
                    low_cond=context.low_cond,
                    high_cond=context.high_cond,
                    maxiter=de_maxiter,
                    popsize=de_popsize,
                    mutation=de_mutation,
                    recombination=de_recombination,
                    seed=de_seed,
                    adjacency=context.adjacency,
                    electrode_supports=context.electrode_supports,
                    enforce_connectivity=enforce_connectivity,
                    repair_disconnected=repair_disconnected,
                    benchmark_name=context.benchmark_name,
                    benchmark_weights=context.benchmark_weights,
                    progress_callback=progress_callback,
                )

                self.optimization_queue.put({"type": "done", "payload": result})
            except Exception as exc:
                self.optimization_queue.put(
                    {
                        "type": "error",
                        "payload": f"{exc}\n\n{traceback.format_exc()}",
                    }
                )

        self.optimization_thread = threading.Thread(target=worker, daemon=True)
        self.optimization_thread.start()

    def request_stop(self) -> None:
        messagebox.showinfo(
            "Stop request",
            "This UI currently runs each optimizer generation to completion.\n"
            "The stop button only prevents new runs from starting until the current one finishes.",
        )

    def _poll_queue(self) -> None:
        try:
            while True:
                item = self.optimization_queue.get_nowait()
                item_type = item.get("type")
                payload = item.get("payload")

                if item_type == "meta":
                    n_params = int(payload.get("n_params", 0))
                    self.param_count_var.set(f"Parameters: {n_params}")
                    self.status_var.set(f"Optimization configured with {n_params} parameters.")
                elif item_type == "progress":
                    self.latest_progress = dict(payload)
                    self._consume_progress(payload)
                elif item_type == "done":
                    self.optimization_result = dict(payload)
                    self.is_running = False
                    self.optimize_button.configure(state=tk.NORMAL)
                    self.stop_button.configure(state=tk.DISABLED)
                    self._finalize_result(payload)
                elif item_type == "error":
                    self.is_running = False
                    self.optimize_button.configure(state=tk.NORMAL)
                    self.stop_button.configure(state=tk.DISABLED)
                    messagebox.showerror("Optimization failed", str(payload))
                    self.status_var.set(f"Optimization failed: {payload}")
                elif item_type == "biomimetic_done":
                    self.biomimetic_status_var.set("Biomimetic analysis complete.")
                    touch_metrics = dict(payload.get("touch_metrics", {}))
                    touch_sensitivity = np.asarray(payload.get("touch_sensitivity", []), dtype=float).ravel()
                    fungal_history = np.asarray(payload.get("fungal_history", []), dtype=float).ravel()
                    fungal_sigma = np.asarray(payload.get("fungal_sigma", []), dtype=float).ravel()
                    self.biomimetic_result_var.set(
                        " | ".join(
                            [
                                f"Touch expected={touch_metrics.get('expected', np.nan):.6e}",
                                f"minimax={touch_metrics.get('minimax', np.nan):.6e}",
                                f"softmin={touch_metrics.get('softmin', np.nan):.6e}",
                                f"SNR={touch_metrics.get('snr', np.nan):.6e}",
                            ]
                        )
                    )
                    self._render_biomimetic_result(touch_sensitivity, fungal_sigma, fungal_history)
                elif item_type == "biomimetic_error":
                    self.biomimetic_status_var.set("Biomimetic analysis failed.")
                    messagebox.showerror("Biomimetic analysis failed", str(payload))
        except queue.Empty:
            pass

        self.root.after(120, self._poll_queue)

    def _consume_progress(self, payload: dict[str, Any]) -> None:
        generation = int(payload.get("generation", 0))
        total = int(payload.get("n_generations", 0))
        best_score = float(payload.get("best_score", np.inf))
        current_score = float(payload.get("current_score", np.inf))
        best_perm = payload.get("best_perm")
        best_state = payload.get("best_state")
        best_sensitivity = payload.get("best_sensitivity")
        best_theta = payload.get("best_theta")

        self.live_best_history.append(best_score)
        self.live_current_history.append(current_score)

        self.generation_var.set(f"Generation: {generation} / {total}")
        self.best_score_var.set(f"Best score: {best_score:.6e}")
        self.current_score_var.set(f"Current score: {current_score:.6e}")
        self.progress_var.set("Updating live mesh from best generation...")

        if best_theta is not None:
            self.param_count_var.set(f"Parameters: {len(np.asarray(best_theta).ravel())}")

        self._render_result_snapshot(
            best_perm=best_perm,
            best_state=best_state,
            best_sensitivity=best_sensitivity,
            history_best=np.asarray(self.live_best_history, dtype=float),
            history_current=np.asarray(self.live_current_history, dtype=float),
        )

    def _finalize_result(self, result: dict[str, Any]) -> None:
        best_score = float(result.get("best_score", np.inf))
        self.best_score_var.set(f"Best score: {best_score:.6e}")
        current_history = np.asarray(result.get("current_history", []), dtype=float).ravel()
        current_value = float(current_history[-1]) if current_history.size else best_score
        self.current_score_var.set(f"Current score: {current_value:.6e}")
        self.progress_var.set("Optimization complete.")
        self.status_var.set("Optimization finished. You can save the full run artifacts now.")
        self.save_info_var.set("Optimization completed. Saving is available.")
        self._render_result_snapshot(
            best_perm=result.get("best_perm"),
            best_state=result.get("best_state"),
            best_sensitivity=result.get("best_sensitivity"),
            history_best=np.asarray(result.get("best_history", []), dtype=float),
            history_current=np.asarray(result.get("current_history", []), dtype=float),
        )

        if self.save_results_var.get():
            try:
                self.save_current_results()
            except Exception as exc:
                messagebox.showerror("Auto-save failed", str(exc))

    def _render_mesh(self) -> None:
        if self.mesh_obj is None:
            self._draw_idle_state()
            return

        pts = np.asarray(self.mesh_obj.node, dtype=float)
        tri = np.asarray(self.mesh_obj.element, dtype=int)
        el_pos = np.asarray(self.mesh_obj.el_pos, dtype=int).reshape(-1)

        self.ax_mesh.clear()
        self.ax_mesh.triplot(pts[:, 0], pts[:, 1], tri, color="#748cab", linewidth=0.6, alpha=0.75)
        if el_pos.size:
            self.ax_mesh.scatter(pts[el_pos, 0], pts[el_pos, 1], c="#d62828", s=32, zorder=3, label="Electrodes")
        self.ax_mesh.set_aspect("equal")
        self.ax_mesh.set_title("Generated mesh")
        self.ax_mesh.set_xlabel("x")
        self.ax_mesh.set_ylabel("y")
        self.ax_mesh.grid(alpha=0.15)
        self.ax_mesh.legend(loc="upper right")

        self.ax_sensitivity.clear()
        self.ax_sensitivity.set_title("Sensitivity preview")
        self.ax_sensitivity.text(0.5, 0.5, "Run optimization to see live sensitivity", ha="center", va="center", transform=self.ax_sensitivity.transAxes)
        self.ax_sensitivity.set_axis_off()

        self.ax_history.clear()
        self.ax_history.set_title("Optimization history")
        self.ax_history.text(0.5, 0.5, "Run optimization to see live scores", ha="center", va="center", transform=self.ax_history.transAxes)
        self.ax_history.set_axis_off()
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _render_result_snapshot(
        self,
        best_perm: Any,
        best_state: Any,
        best_sensitivity: Any,
        history_best: np.ndarray | None,
        history_current: np.ndarray | None,
    ) -> None:
        if self.mesh_obj is None:
            return

        pts = np.asarray(self.mesh_obj.node, dtype=float)
        tri = np.asarray(self.mesh_obj.element, dtype=int)
        el_pos = np.asarray(self.mesh_obj.el_pos, dtype=int).reshape(-1)

        if self.mesh_colorbar is not None:
            self.mesh_colorbar.remove()
            self.mesh_colorbar = None
        if self.sensitivity_colorbar is not None:
            self.sensitivity_colorbar.remove()
            self.sensitivity_colorbar = None

        self.ax_mesh.clear()
        if best_perm is not None:
            perm = np.asarray(best_perm, dtype=float).ravel()
            if perm.size == tri.shape[0]:
                im = self.ax_mesh.tripcolor(pts[:, 0], pts[:, 1], tri, perm, shading="flat", cmap="RdBu_r")
                self.mesh_colorbar = self.figure.colorbar(im, ax=self.ax_mesh, fraction=0.046, pad=0.04, label="Conductivity")
            else:
                self.ax_mesh.triplot(pts[:, 0], pts[:, 1], tri, color="#748cab", linewidth=0.6, alpha=0.75)
        else:
            self.ax_mesh.triplot(pts[:, 0], pts[:, 1], tri, color="#748cab", linewidth=0.6, alpha=0.75)

        if el_pos.size:
            self.ax_mesh.scatter(pts[el_pos, 0], pts[el_pos, 1], c="#111111", s=24, zorder=3, label="Electrodes")
        self.ax_mesh.set_aspect("equal")
        self.ax_mesh.set_title("Live best mesh")
        self.ax_mesh.set_xlabel("x")
        self.ax_mesh.set_ylabel("y")
        self.ax_mesh.grid(alpha=0.15)
        self.ax_mesh.legend(loc="upper right")

        self.ax_sensitivity.clear()
        display_baseline = self.metric_option_var.get().strip().lower() == "uniformity" and self.reference_sensitivity is not None
        source_sensitivity = self.reference_sensitivity if display_baseline else best_sensitivity

        if source_sensitivity is not None:
            sens = np.asarray(source_sensitivity, dtype=float).ravel()
            if sens.size == tri.shape[0]:
                sens_display = sensitivity_for_display(sens)
                im_sens = self.ax_sensitivity.tripcolor(pts[:, 0], pts[:, 1], tri, sens_display, shading="flat", cmap="viridis")
                self.sensitivity_colorbar = self.figure.colorbar(im_sens, ax=self.ax_sensitivity, fraction=0.046, pad=0.04, label="log10(sum |J[:, elem]|)")
            else:
                self.ax_sensitivity.triplot(pts[:, 0], pts[:, 1], tri, color="#748cab", linewidth=0.6, alpha=0.75)
        else:
            self.ax_sensitivity.triplot(pts[:, 0], pts[:, 1], tri, color="#748cab", linewidth=0.6, alpha=0.75)

        if el_pos.size:
            self.ax_sensitivity.scatter(pts[el_pos, 0], pts[el_pos, 1], c="#111111", s=24, zorder=3, label="Electrodes")
        self.ax_sensitivity.set_aspect("equal")
        self.ax_sensitivity.set_title(
            "Reference sensitivity (uniform perm, log10 display)"
            if display_baseline
            else "Live sensitivity (log10 display)"
        )
        self.ax_sensitivity.set_xlabel("x")
        self.ax_sensitivity.set_ylabel("y")
        self.ax_sensitivity.grid(alpha=0.15)
        self.ax_sensitivity.legend(loc="upper right")

        self.ax_history.clear()
        if history_best is not None and history_best.size:
            self.ax_history.plot(history_best, label="Best so far", linewidth=1.8, color="#003049")
        if history_current is not None and history_current.size:
            self.ax_history.plot(history_current, label="Current generation", linewidth=1.1, alpha=0.75, color="#d62828")
        self.ax_history.set_title("Optimization history")
        self.ax_history.set_xlabel("Generation")
        self.ax_history.set_ylabel("Score")
        self.ax_history.grid(alpha=0.25)
        if history_best is not None and history_best.size or history_current is not None and history_current.size:
            self.ax_history.legend(loc="best")
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _render_biomimetic_result(self, touch_sensitivity: np.ndarray, fungal_sigma: np.ndarray, fungal_history: np.ndarray) -> None:
        if self.mesh_obj is None:
            return

        pts = np.asarray(self.mesh_obj.node, dtype=float)
        tri = np.asarray(self.mesh_obj.element, dtype=int)
        el_pos = np.asarray(self.mesh_obj.el_pos, dtype=int).reshape(-1)

        if self.mesh_colorbar is not None:
            self.mesh_colorbar.remove()
            self.mesh_colorbar = None
        if self.sensitivity_colorbar is not None:
            self.sensitivity_colorbar.remove()
            self.sensitivity_colorbar = None

        self.ax_mesh.clear()
        if fungal_sigma.size == tri.shape[0]:
            im = self.ax_mesh.tripcolor(pts[:, 0], pts[:, 1], tri, fungal_sigma, shading="flat", cmap="cividis")
            self.mesh_colorbar = self.figure.colorbar(im, ax=self.ax_mesh, fraction=0.046, pad=0.04, label="Fungal sigma")
        else:
            self.ax_mesh.triplot(pts[:, 0], pts[:, 1], tri, color="#748cab", linewidth=0.6, alpha=0.75)
        if el_pos.size:
            self.ax_mesh.scatter(pts[el_pos, 0], pts[el_pos, 1], c="#111111", s=24, zorder=3, label="Electrodes")
        self.ax_mesh.set_aspect("equal")
        self.ax_mesh.set_title("Fungal growth conductivity")
        self.ax_mesh.set_xlabel("x")
        self.ax_mesh.set_ylabel("y")
        self.ax_mesh.grid(alpha=0.15)
        self.ax_mesh.legend(loc="upper right")

        self.ax_sensitivity.clear()
        if touch_sensitivity.size == tri.shape[0]:
            touch_display = sensitivity_for_display(touch_sensitivity)
            im_touch = self.ax_sensitivity.tripcolor(pts[:, 0], pts[:, 1], tri, touch_display, shading="flat", cmap="viridis")
            self.sensitivity_colorbar = self.figure.colorbar(im_touch, ax=self.ax_sensitivity, fraction=0.046, pad=0.04, label="log10(sum |J[:, elem]|)")
        else:
            self.ax_sensitivity.triplot(pts[:, 0], pts[:, 1], tri, color="#748cab", linewidth=0.6, alpha=0.75)
        if el_pos.size:
            self.ax_sensitivity.scatter(pts[el_pos, 0], pts[el_pos, 1], c="#111111", s=24, zorder=3, label="Electrodes")
        self.ax_sensitivity.set_aspect("equal")
        self.ax_sensitivity.set_title("Touch sensitivity proxy (log10 display)")
        self.ax_sensitivity.set_xlabel("x")
        self.ax_sensitivity.set_ylabel("y")
        self.ax_sensitivity.grid(alpha=0.15)
        self.ax_sensitivity.legend(loc="upper right")

        self.ax_history.clear()
        if fungal_history.size:
            self.ax_history.plot(fungal_history, label="Fungal cost", linewidth=1.8, color="#003049")
        self.ax_history.set_title("Biomimetic history")
        self.ax_history.set_xlabel("Growth iteration")
        self.ax_history.set_ylabel("Cost")
        self.ax_history.grid(alpha=0.25)
        if fungal_history.size:
            self.ax_history.legend(loc="best")
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def browse_output_dir(self) -> None:
        path = filedialog.askdirectory(title="Select output directory")
        if path:
            self.output_dir_var.set(path)

    def _current_run_name(self) -> str:
        name = str(self.run_name_var.get()).strip()
        return name or _build_run_name(SimpleNamespace(optimizer="de", benchmark_name="sensitivity_uniformity", run_name=""))

    def _build_save_args(self) -> SimpleNamespace:
        return SimpleNamespace(
            optimizer="de",
            benchmark_name=self._selected_benchmark_name(),
            benchmark_profile=self._selected_benchmark_name(),
            seed=parse_int(self.seed_var, "seed"),
            h0=parse_float(self.h0_var, "mesh h0"),
            low_cond=parse_float(self.low_cond_var, "low conductivity"),
            high_cond=parse_float(self.high_cond_var, "high conductivity"),
            output_dir=str(Path(self.output_dir_var.get()).expanduser()),
            run_name=self._current_run_name(),
        )

    def save_current_results(self) -> None:
        if self.context is None or self.optimization_result is None or self.mesh_obj is None:
            messagebox.showwarning("Nothing to save", "Run an optimization before saving results.")
            return

        args = self._build_save_args()
        connectivity_info = self._connectivity_summary(self.optimization_result.get("best_state"))
        artifact_paths = save_run_artifacts(
            output_dir=args.output_dir,
            run_name=args.run_name,
            mesh_obj=self.mesh_obj,
            p1=self.context.p1,
            p2=self.context.p2,
            result=self.optimization_result,
            args=args,
            field_shape_cfg=self.context.field_shape_cfg,
            benchmark_weights=self.context.benchmark_weights,
            connectivity_info=connectivity_info,
        )
        self.last_saved_npz_path = artifact_paths["npz"]
        self.benchmark_npz_var.set(artifact_paths["npz"])
        default_score_json = str(Path(artifact_paths["npz"]).with_name(Path(artifact_paths["npz"]).stem + "_benchmark.json"))
        if not self.benchmark_score_json_var.get().strip():
            self.benchmark_score_json_var.set(default_score_json)
        self.save_info_var.set(f"Saved artifacts to {artifact_paths['npz']}")
        messagebox.showinfo(
            "Save complete",
            "Saved run artifacts:\n"
            f"NPZ: {artifact_paths['npz']}\n"
            f"JSON: {artifact_paths['metadata_json']}\n"
            f"History CSV: {artifact_paths['history_csv']}\n"
            f"Elements CSV: {artifact_paths['elements_csv']}",
        )

    def open_detailed_result_plot(self) -> None:
        if self.context is None or self.optimization_result is None or self.mesh_obj is None:
            messagebox.showwarning("No result", "Run an optimization first.")
            return

        try:
            plot_results(
                mesh_obj=self.mesh_obj,
                p1=self.context.p1,
                p2=self.context.p2,
                best_perm=self.optimization_result["best_perm"],
                best_sensitivity=self.optimization_result["best_sensitivity"],
                best_hist=self.optimization_result["best_history"],
                curr_hist=self.optimization_result["current_history"],
                best_score=self.optimization_result["best_score"],
                best_theta=self.optimization_result.get("best_theta"),
                field_shape_cfg=self.context.field_shape_cfg,
                comparison_rows=[],
            )
        except Exception as exc:
            messagebox.showerror("Plot failed", str(exc))

    def browse_benchmark_npz(self) -> None:
        path = filedialog.askopenfilename(
            title="Select saved optimization NPZ",
            filetypes=[("NPZ files", "*.npz"), ("All files", "*.*")],
        )
        if path:
            self.benchmark_npz_var.set(path)

    def load_last_saved_run_for_benchmark(self) -> None:
        if not self.last_saved_npz_path:
            messagebox.showinfo("No saved run", "Save an optimization result first.")
            return
        self.benchmark_npz_var.set(self.last_saved_npz_path)
        self.benchmark_result_var.set(f"Loaded last saved run: {self.last_saved_npz_path}")

    def use_current_saved_run_for_benchmark(self) -> None:
        if self.last_saved_npz_path:
            self.benchmark_npz_var.set(self.last_saved_npz_path)
        elif self.optimization_result is not None and self.context is not None:
            self.save_current_results()
        else:
            messagebox.showinfo("No saved run", "Run and save an optimization first.")

    def run_saved_benchmark(self) -> None:
        npz_path = str(self.benchmark_npz_var.get()).strip()
        if not npz_path:
            messagebox.showwarning("Missing NPZ", "Select a saved NPZ file first.")
            return

        try:
            benchmark_name = self._selected_benchmark_name()
            benchmark_weights = self._build_benchmark_weights()
            result = benchmark_saved_result(
                npz_path=npz_path,
                benchmark_name=benchmark_name,
                benchmark_weights=benchmark_weights,
            )

            self.benchmark_result_var.set(
                f"Score={result['score']:.6e} | connected={result['high_elements_connected']} | isolated={result['isolated_high_elements']}"
            )

            score_json = str(self.benchmark_score_json_var.get()).strip()
            if score_json:
                with open(score_json, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2)
            messagebox.showinfo(
                "Benchmark complete",
                "Saved benchmark summary:\n"
                f"Score: {result['score']:.6e}\n"
                f"Connected: {result['high_elements_connected']}\n"
                f"Isolated high elements: {result['isolated_high_elements']}\n"
                f"Elements: {result['n_elements']}\n"
                f"Source: {result['source_npz']}",
            )
        except Exception as exc:
            messagebox.showerror("Benchmark failed", str(exc))

    def _connectivity_summary(self, state: Any) -> dict[str, Any]:
        if self.context is None or state is None:
            return {
                "is_connected": False,
                "electrodes_linked": False,
                "components": 0,
                "isolated_count": 0,
            }

        state_arr = np.asarray(state, dtype=bool)
        adjacency = self.context.adjacency
        electrode_supports = self.context.electrode_supports
        components = get_connected_components(state_arr, adjacency)
        return {
            "is_connected": bool(has_connected_high_elements(state_arr, adjacency)),
            "electrodes_linked": bool(electrodes_connected_by_high_region(state_arr, adjacency, electrode_supports)),
            "components": int(len(components)),
            "isolated_count": int(count_isolated_high_elements(state_arr, adjacency)),
        }


def main() -> None:
    plt.close("all")
    root = tk.Tk()
    CombinedOptimizerUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
