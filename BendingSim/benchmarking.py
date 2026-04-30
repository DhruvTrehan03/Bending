"""Standalone benchmarking and scoring helpers."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import deque
import numpy as np
import matplotlib.pyplot as plt

try:
    from biomimetic_optimizers import TouchSensitivityCost
except ImportError:
    from .biomimetic_optimizers import TouchSensitivityCost


BenchmarkRegistry = {}


def register_benchmark(name):
    """Register a benchmark scorer by name."""

    def _decorate(fn):
        BenchmarkRegistry[str(name).strip().lower()] = fn
        return fn

    return _decorate


def list_benchmarks():
    return sorted(BenchmarkRegistry.keys())


def get_benchmark(name):
    key = str(name).strip().lower()
    if key not in BenchmarkRegistry:
        raise ValueError(
            f"Unknown benchmark '{name}'. Available: {', '.join(list_benchmarks())}"
        )
    return BenchmarkRegistry[key]


BENCHMARK_PROFILES = {
    "uniformity": {
        "entropy_weight": 0.5,
        "uniformity_weight": 1.0,
        "isolated_penalty": 2.0,
        "disconnected_penalty": 20.0,
    },
}


def get_profile_weights(profile_name, overrides=None):
    profile_key = str(profile_name).strip().lower()
    if profile_key not in BENCHMARK_PROFILES:
        raise ValueError(
            f"Unknown benchmark profile '{profile_name}'. "
            f"Available: {', '.join(sorted(BENCHMARK_PROFILES.keys()))}"
        )
    weights = dict(BENCHMARK_PROFILES[profile_key])
    if overrides:
        weights.update(overrides)
    return weights


def make_permittivity(state_high, variable_mask, low_cond, high_cond):
    n_elem = variable_mask.size
    perm = np.full(n_elem, low_cond, dtype=float)

    variable_indices = np.where(variable_mask)[0]
    perm[variable_indices[state_high]] = high_cond
    return perm


def expand_state_to_full_mesh(state_high_var, variable_mask):
    """Expand state from variable-element space to full-element space.
    
    When working with element models, the state is indexed by variable elements only.
    This function expands it to include all elements (False for non-variable).
    
    Args:
        state_high_var: Bool array of size = number of variable elements
        variable_mask: Bool array of size = number of total elements
    
    Returns:
        Bool array of size = number of total elements
    """
    state_high_var = np.asarray(state_high_var, dtype=bool)
    variable_mask = np.asarray(variable_mask, dtype=bool)
    
    n_total = variable_mask.size
    n_var = np.sum(variable_mask)
    
    if state_high_var.size != n_var:
        raise ValueError(
            f"state_high_var size {state_high_var.size} does not match "
            f"variable elements count {n_var} from variable_mask"
        )
    
    state_full = np.zeros(n_total, dtype=bool)
    variable_indices = np.where(variable_mask)[0]
    state_full[variable_indices] = state_high_var
    return state_full


def get_state_for_connectivity(state_high, variable_mask):
    """Get state properly sized for connectivity functions.
    
    Ensures state represents all elements (expanding from variable-element space if needed).
    
    Args:
        state_high: Bool array (either variable-element indexed or full-mesh indexed)
        variable_mask: Bool array of size = number of total elements
    
    Returns:
        Bool array of size = number of total elements
    """
    state_high = np.asarray(state_high, dtype=bool)
    variable_mask = np.asarray(variable_mask, dtype=bool)
    
    n_total = variable_mask.size
    n_var = np.sum(variable_mask)
    
    # If state matches total elements, return as-is
    if state_high.size == n_total:
        return state_high
    
    # If state matches variable elements, expand it
    if state_high.size == n_var:
        return expand_state_to_full_mesh(state_high, variable_mask)
    
    raise ValueError(
        f"state_high size {state_high.size} does not match "
        f"total elements {n_total} or variable elements {n_var}"
    )


def element_sensitivity_from_jacobian(jacobian):
    return np.log10(np.sum(np.abs(jacobian), axis=0) + np.finfo(float).tiny)


def _touch_cost_from_context(jacobian, mesh_centroids, weights=None):
    if mesh_centroids is None:
        raise ValueError("Touch sensitivity benchmarks require mesh_centroids.")

    local_weights = dict(weights or {})
    touch_radius = float(local_weights.get("touch_radius", 0.30))
    touch_noise_scale = local_weights.get("touch_noise_scale", None)
    touch_samples = int(max(1, local_weights.get("touch_samples", 200)))
    touch_temperature = float(local_weights.get("touch_temperature", 1.0))
    touch_lambda = float(local_weights.get("touch_lambda", 0.1))
    touch_seed = local_weights.get("touch_seed", None)

    noise_cov = None
    if touch_noise_scale is not None:
        noise_scale = float(touch_noise_scale)
        if noise_scale > 0.0:
            n_measurements = int(np.asarray(jacobian).shape[0])
            noise_cov = np.eye(n_measurements, dtype=float) * (noise_scale ** 2)

    return TouchSensitivityCost(
        J=np.asarray(jacobian, dtype=float),
        mesh_elements=np.asarray(mesh_centroids, dtype=float),
        touch_radius=touch_radius,
        noise_cov=noise_cov,
        rng=touch_seed,
    ), touch_samples, touch_temperature, touch_lambda


def entropy_score(sensitivity):
    vals = np.abs(np.asarray(sensitivity).ravel())
    eps = np.finfo(float).eps

    total = np.sum(vals)
    if total < eps:
        return 0.0

    probabilities = vals / total
    entropy = -np.sum(probabilities[probabilities > eps] * np.log(probabilities[probabilities > eps]))
    mean_sensitivity = np.mean(vals)
    return float(entropy + mean_sensitivity)


def get_connected_components(state_high, adjacency):
    n_elem = len(state_high)
    visited = np.zeros(n_elem, dtype=bool)
    components = []

    for start_elem in range(n_elem):
        if not state_high[start_elem] or visited[start_elem]:
            continue

        component = []
        queue = [start_elem]
        visited[start_elem] = True

        while queue:
            elem_idx = queue.pop(0)
            component.append(elem_idx)
            for adjacent_elem in adjacency[elem_idx]:
                if not visited[adjacent_elem] and state_high[adjacent_elem]:
                    visited[adjacent_elem] = True
                    queue.append(adjacent_elem)

        components.append(component)

    return components


def count_isolated_high_elements(state_high, adjacency):
    isolated_count = 0
    for i in range(len(state_high)):
        if state_high[i]:
            has_high_neighbor = any(state_high[j] for j in adjacency[i])
            if not has_high_neighbor:
                isolated_count += 1
    return isolated_count


def repair_disconnected_state(state_high, adjacency):
    """Repair a disconnected state by connecting all components.
    
    Finds disconnected components and bridges them using a greedy approach:
    finds the shortest path between each pair of components and marks
    elements along that path as high.
    
    Args:
        state_high: Bool array indicating high conductivity elements
        adjacency: Adjacency list for elements
    
    Returns:
        Repaired bool array with all components connected
    """
    state_high = np.asarray(state_high, dtype=bool).copy()
    n_high = np.sum(state_high)
    
    if n_high <= 1:
        return state_high  # Already valid
    
    components = get_connected_components(state_high, adjacency)
    if len(components) == 1:
        return state_high  # Already connected
    
    # Connect components greedily
    # Start with the first component and connect others to it
    connected_component = set(components[0])
    
    for component in components[1:]:
        # Find shortest path from connected_component to this component
        component_set = set(component)
        shortest_path = None
        shortest_dist = float('inf')
        
        # BFS to find shortest path between two sets
        queue = deque([(elem, 0, []) for elem in connected_component])
        visited = set(connected_component)
        
        while queue and shortest_dist == float('inf'):
            elem, dist, path = queue.popleft()
            
            for neighbor in adjacency[elem]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_path = path + [neighbor]
                    
                    if neighbor in component_set:
                        # Found target
                        if dist + 1 < shortest_dist:
                            shortest_dist = dist + 1
                            shortest_path = new_path
                        break
                    else:
                        # Continue searching
                        queue.append((neighbor, dist + 1, new_path))
        
        # Add path elements to state
        if shortest_path is not None:
            for elem_idx in shortest_path:
                state_high[elem_idx] = True
            # Add component to connected set
            connected_component.update(component_set)
    
    return state_high


def has_connected_high_elements(state_high, adjacency):
    n_high = np.sum(state_high)
    if n_high <= 1:
        return True

    components = get_connected_components(state_high, adjacency)
    return len(components) == 1


def electrodes_connected_by_high_region(state_high, adjacency, electrode_supports):
    if electrode_supports is None:
        return has_connected_high_elements(state_high, adjacency)

    components = get_connected_components(state_high, adjacency)
    if len(components) != 1:
        return False

    for support in electrode_supports:
        support = np.asarray(support, dtype=int)
        if support.size == 0:
            return False
        if not np.any(state_high[support]):
            return False

    return True


def validate_high_element_connectivity(state_high, adjacency, electrode_supports=None, strict=True, variable_mask=None):
    """Validate that all high conductivity elements are properly connected.
    
    Args:
        state_high: Bool array indicating high conductivity elements
                   (can be variable-element indexed or full-mesh indexed)
        adjacency: Adjacency list for elements (full-mesh sized)
        electrode_supports: Optional list of electrode support element indices
        strict: If True, require all electrodes connected; if False, just check element connectivity
        variable_mask: Optional bool array to indicate which elements are variable (enables state expansion)
    
    Returns:
        Dict with connectivity status and metrics
    """
    # Ensure state is properly sized for full-mesh adjacency
    if variable_mask is not None:
        state_high = get_state_for_connectivity(state_high, variable_mask)
    
    state_high = np.asarray(state_high, dtype=bool)
    n_high = np.sum(state_high)
    
    if n_high == 0:
        return {
            "valid": False,
            "reason": "No high conductivity elements",
            "n_high": 0,
            "n_components": 0,
            "isolated_count": 0,
            "all_electrodes_connected": False,
        }
    
    if n_high == 1:
        isolated = count_isolated_high_elements(state_high, adjacency)
        return {
            "valid": isolated == 0,
            "reason": "Single element" if isolated == 0 else "Single isolated element",
            "n_high": 1,
            "n_components": 1,
            "isolated_count": isolated,
            "all_electrodes_connected": not strict or (electrode_supports and _all_electrodes_have_high(state_high, electrode_supports)),
        }
    
    components = get_connected_components(state_high, adjacency)
    isolated = count_isolated_high_elements(state_high, adjacency)
    
    result = {
        "valid": len(components) == 1 and isolated == 0,
        "n_high": int(n_high),
        "n_components": len(components),
        "isolated_count": isolated,
        "all_electrodes_connected": False,
    }
    
    if len(components) > 1:
        result["reason"] = f"Disconnected: {len(components)} components"
    elif isolated > 0:
        result["reason"] = f"{isolated} isolated elements"
    else:
        result["reason"] = "Connected"
    
    if electrode_supports is not None:
        all_elecs_connected = electrodes_connected_by_high_region(state_high, adjacency, electrode_supports)
        result["all_electrodes_connected"] = all_elecs_connected
        if strict and not all_elecs_connected:
            result["valid"] = False
            result["reason"] = "Not all electrodes connected to high region"
    
    return result


def _all_electrodes_have_high(state_high, electrode_supports):
    """Check if all electrode supports have at least one high conductivity element."""
    if electrode_supports is None:
        return True
    for support in electrode_supports:
        support = np.asarray(support, dtype=int)
        if support.size == 0 or not np.any(state_high[support]):
            return False
    return True


def benchmark_score(sensitivity, state, adjacency=None, weights=None, **_ignored):
    """Task-specific benchmark score.

    Weights can include:
    - entropy
    - isolated_penalty
    - disconnected_penalty
    """
    if weights is None:
        weights = {
            "entropy": 1.0,
            "isolated_penalty": 0.2,
            "disconnected_penalty": 5.0,
        }

    score = float(weights.get("entropy", 1.0)) * entropy_score(sensitivity)

    if adjacency is not None:
        isolated = count_isolated_high_elements(state, adjacency)
        disconnected = 0.0 if has_connected_high_elements(state, adjacency) else 1.0
        score += float(weights.get("isolated_penalty", 0.0)) * float(isolated)
        score += float(weights.get("disconnected_penalty", 0.0)) * float(disconnected)

    return float(score)


@register_benchmark("uniformity")
def benchmark_uniformity(sensitivity, state, adjacency=None, weights=None, **_ignored):
    """Maximize uniformity of sensitivity using entropy-based metric.
    
    Combines entropy and uniformity of sensitivity to ensure both:
    - High information content (entropy)
    - Uniform distribution (low coefficient of variation)
    
    Ensures all high conductivity elements are connected.
    Returns a negative score so that minimization maximizes uniformity + entropy.
    """
    # Get default weights for uniformity profile
    local_weights = get_profile_weights("uniformity", overrides=weights)
    
    # Calculate entropy-based uniformity
    entropy_val = entropy_score(sensitivity)
    uniformity_val = _safe_uniformity(sensitivity)
    
    entropy_weight = float(local_weights.get("entropy_weight", 0.5))
    uniformity_weight = float(local_weights.get("uniformity_weight", 1.0))
    
    # Combine entropy and uniformity (both higher is better, so return negative)
    combined_score = entropy_weight * entropy_val + uniformity_weight * uniformity_val
    score = -combined_score
    
    # Apply strict connectivity constraints if available
    if adjacency is not None:
        isolated_penalty = float(local_weights.get("isolated_penalty", 2.0))
        disconnected_penalty = float(local_weights.get("disconnected_penalty", 20.0))
        
        isolated = count_isolated_high_elements(state, adjacency)
        disconnected = 0.0 if has_connected_high_elements(state, adjacency) else 1.0
        
        score += isolated_penalty * float(isolated)
        score += disconnected_penalty * float(disconnected)
    
    return float(score)


def _touch_metric_score(metric_name, sensitivity, state, adjacency=None, weights=None, jacobian=None, mesh_centroids=None, **_ignored):
    del sensitivity, state, adjacency
    touch_cost, touch_samples, touch_temperature, touch_lambda = _touch_cost_from_context(
        jacobian=jacobian,
        mesh_centroids=mesh_centroids,
        weights=weights,
    )

    metric = str(metric_name).strip().lower()
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
    return _touch_metric_score(
        "expected_sensitivity",
        sensitivity=sensitivity,
        state=state,
        adjacency=adjacency,
        weights=weights,
        jacobian=jacobian,
        mesh_centroids=mesh_centroids,
    )


@register_benchmark("minimax_sensitivity")
def benchmark_minimax_sensitivity(sensitivity, state, adjacency=None, weights=None, jacobian=None, mesh_centroids=None, **_ignored):
    return _touch_metric_score(
        "minimax_sensitivity",
        sensitivity=sensitivity,
        state=state,
        adjacency=adjacency,
        weights=weights,
        jacobian=jacobian,
        mesh_centroids=mesh_centroids,
    )


@register_benchmark("softmin_sensitivity")
def benchmark_softmin_sensitivity(sensitivity, state, adjacency=None, weights=None, jacobian=None, mesh_centroids=None, **_ignored):
    return _touch_metric_score(
        "softmin_sensitivity",
        sensitivity=sensitivity,
        state=state,
        adjacency=adjacency,
        weights=weights,
        jacobian=jacobian,
        mesh_centroids=mesh_centroids,
    )


@register_benchmark("snr_sensitivity")
def benchmark_snr_sensitivity(sensitivity, state, adjacency=None, weights=None, jacobian=None, mesh_centroids=None, **_ignored):
    return _touch_metric_score(
        "snr_sensitivity",
        sensitivity=sensitivity,
        state=state,
        adjacency=adjacency,
        weights=weights,
        jacobian=jacobian,
        mesh_centroids=mesh_centroids,
    )


@register_benchmark("distinguishability")
def benchmark_distinguishability(sensitivity, state, adjacency=None, weights=None, jacobian=None, mesh_centroids=None, **_ignored):
    return _touch_metric_score(
        "distinguishability",
        sensitivity=sensitivity,
        state=state,
        adjacency=adjacency,
        weights=weights,
        jacobian=jacobian,
        mesh_centroids=mesh_centroids,
    )


@register_benchmark("combined")
def benchmark_combined(sensitivity, state, adjacency=None, weights=None, jacobian=None, mesh_centroids=None, **_ignored):
    return _touch_metric_score(
        "combined",
        sensitivity=sensitivity,
        state=state,
        adjacency=adjacency,
        weights=weights,
        jacobian=jacobian,
        mesh_centroids=mesh_centroids,
    )


def evaluate_state(
    fwd,
    state,
    variable_mask,
    low_cond,
    high_cond,
    adjacency=None,
    electrode_supports=None,
    enforce_connectivity=True,
    repair_disconnected=False,
    benchmark_fn=None,
    benchmark_name=None,
    benchmark_weights=None,
    mesh_centroids=None,
):
    """Evaluate a state configuration.
    
    Args:
        fwd: EIT forward model
        state: Element state (variable-element indexed or full-mesh indexed)
        variable_mask: Mask of variable elements
        low_cond: Low conductivity value
        high_cond: High conductivity value
        adjacency: Element adjacency list for connectivity checks
        electrode_supports: Electrode support elements
        enforce_connectivity: If True with repair_disconnected=False, reject disconnected states
        repair_disconnected: If True, repair disconnected states instead of rejecting them (constrains search space)
        benchmark_fn: Custom benchmark function
        benchmark_name: Benchmark name to look up
        benchmark_weights: Benchmark weights dict
    
    Returns:
        (perm, sensitivity, score) tuple
    """
    # Ensure state is properly sized for full-mesh operations
    if variable_mask is not None:
        state_full = get_state_for_connectivity(state, variable_mask)
    else:
        state_full = np.asarray(state, dtype=bool)
    
    # Apply connectivity repair if enabled (constrains search space to connected solutions only)
    if repair_disconnected and adjacency is not None:
        state_full = repair_disconnected_state(state_full, adjacency)
    
    perm = make_permittivity(
        state_high=state_full,
        variable_mask=variable_mask,
        low_cond=low_cond,
        high_cond=high_cond,
    )

    # Hard reject only if enforce_connectivity=True and repair_disconnected=False
    if enforce_connectivity and not repair_disconnected and adjacency is not None:
        if not electrodes_connected_by_high_region(state_full, adjacency, electrode_supports):
            sensitivity = np.zeros_like(perm, dtype=float)
            return perm, sensitivity, np.inf

    jacobian, _ = fwd.compute_jac(perm=perm)
    sensitivity = element_sensitivity_from_jacobian(jacobian)

    if benchmark_fn is None:
        if benchmark_name is None:
            benchmark_fn = benchmark_score
        else:
            benchmark_fn = get_benchmark(benchmark_name)

    score = float(
        benchmark_fn(
            sensitivity=sensitivity,
            state=state_full,
            adjacency=adjacency,
            weights=benchmark_weights,
            jacobian=jacobian,
            mesh_centroids=mesh_centroids,
        )
    )

    return perm, sensitivity, float(score)


def _safe_uniformity(values):
    vals = np.abs(np.asarray(values, dtype=float).ravel())
    eps = np.finfo(float).eps
    mean_val = float(np.mean(vals))
    if mean_val <= eps:
        return 0.0
    cv = float(np.std(vals) / (mean_val + eps))
    return float(1.0 / (1.0 + cv))


def _masked_centroid(points, weights):
    w = np.asarray(weights, dtype=float).ravel()
    pts = np.asarray(points, dtype=float)
    eps = np.finfo(float).eps
    sw = float(np.sum(w))
    if sw <= eps:
        return np.mean(pts, axis=0)
    return np.sum(pts * w[:, None], axis=0) / (sw + eps)


def _prediction_mask_from_reconstruction(ds, n_true):
    n_true = int(max(1, n_true))
    values = np.abs(np.asarray(ds, dtype=float).ravel())
    order = np.argsort(values)[::-1]
    keep = order[:n_true]
    mask = np.zeros(values.size, dtype=bool)
    mask[keep] = True
    return mask


def _dice_score(mask_true, mask_pred):
    t = np.asarray(mask_true, dtype=bool)
    p = np.asarray(mask_pred, dtype=bool)
    tp = np.count_nonzero(t & p)
    denom = (2 * tp) + np.count_nonzero(t & ~p) + np.count_nonzero(~t & p)
    if denom == 0:
        return 0.0
    return float((2.0 * tp) / float(denom))


def _domain_centers_for_radius(mesh_obj, radius, count, margin_frac=0.05):
    pts = np.asarray(mesh_obj.node, dtype=float)
    x_min = float(np.min(pts[:, 0]))
    x_max = float(np.max(pts[:, 0]))
    y_min = float(np.min(pts[:, 1]))
    y_max = float(np.max(pts[:, 1]))
    width = max(np.finfo(float).eps, x_max - x_min)
    height = max(np.finfo(float).eps, y_max - y_min)
    margin = float(max(0.0, margin_frac)) * min(width, height)

    x_lo = x_min + radius + margin
    x_hi = x_max - radius - margin
    y_lo = y_min + radius + margin
    y_hi = y_max - radius - margin

    if x_lo >= x_hi or y_lo >= y_hi:
        return np.asarray([[(x_min + x_max) * 0.5, (y_min + y_max) * 0.5]], dtype=float)

    count = int(max(1, count))
    nx = int(np.ceil(np.sqrt(count)))
    ny = int(np.ceil(count / nx))
    xs = np.linspace(x_lo, x_hi, nx)
    ys = np.linspace(y_lo, y_hi, ny)

    centers = []
    for y in ys:
        for x in xs:
            centers.append([x, y])
            if len(centers) >= count:
                return np.asarray(centers, dtype=float)
    return np.asarray(centers, dtype=float)


def _circle_mask_from_centroids(centroids, center, radius):
    ctr = np.asarray(center, dtype=float)
    dist = np.sqrt(np.sum((np.asarray(centroids, dtype=float) - ctr[None, :]) ** 2, axis=1))
    mask = dist <= float(radius)
    if not np.any(mask):
        mask[np.argmin(dist)] = True
    return mask


def run_circular_anomaly_benchmark(
    n_el=16,
    h0=0.08,
    radius_values=None,
    centers_per_radius=9,
    anomaly_delta=8.0,
    background_cond=1.0,
    jac_p=0.20,
    jac_lamb=0.001,
    objective_weights=None,
):
    """Evaluate anomaly-detection benchmarks for circular anomalies.

    Metrics per anomaly case:
    - signal_change: relative L2 voltage difference ||v1-v0|| / ||v0||
    - sensitivity_uniformity: 1 / (1 + coefficient_of_variation)
    - sensitivity_strength: mean local sensitivity / global mean sensitivity
    - reconstruction: Dice overlap between true anomaly and top-k reconstruction map
    - current_distribution: uniformity of measurement-path current proxy
    """
    import pyeit.eit.protocol as protocol
    import pyeit.mesh as mesh
    from pyeit.eit.fem import EITForward
    from pyeit.eit.jac import JAC
    from pyeit.mesh.shape import circle

    if radius_values is None:
        radius_values = np.array([0.10, 0.16, 0.22, 0.30], dtype=float)
    else:
        radius_values = np.asarray(radius_values, dtype=float).ravel()

    mesh_obj = mesh.create(
        n_el=int(n_el),
        h0=float(h0),
        fd=circle,
        bbox=[[-1.0, -1.0], [1.0, 1.0]],
    )
    proto = protocol.create(int(n_el), dist_exc=max(1, int(n_el) // 2), step_meas=1, parser_meas="rotate_meas")
    fwd = EITForward(mesh_obj, proto)

    jac_solver = JAC(mesh_obj, proto)
    jac_solver.setup(p=float(jac_p), lamb=float(jac_lamb), method="kotre")

    tri = np.asarray(mesh_obj.element)
    pts = np.asarray(mesh_obj.node)
    centroids = np.mean(pts[tri], axis=1)[:, :2]
    n_elem = tri.shape[0]

    perm0 = np.full(n_elem, float(background_cond), dtype=float)
    v0 = np.asarray(fwd.solve_eit(perm=perm0), dtype=float).ravel()

    default_weights = {
        "signal_change": 1.0,
        "sensitivity_uniformity": 0.5,
        "sensitivity_strength": 0.8,
        "reconstruction": 1.0,
        "current_distribution": 0.6,
    }
    if objective_weights:
        default_weights.update(objective_weights)
    weights = default_weights

    cases = []
    eps = np.finfo(float).eps
    domain_diag = float(np.sqrt((pts[:, 0].max() - pts[:, 0].min()) ** 2 + (pts[:, 1].max() - pts[:, 1].min()) ** 2))
    domain_diag = max(domain_diag, eps)

    for radius in radius_values:
        centers = _domain_centers_for_radius(
            mesh_obj=mesh_obj,
            radius=float(radius),
            count=int(centers_per_radius),
        )
        for center in centers:
            in_circle = _circle_mask_from_centroids(centroids, center, radius)
            if not np.any(in_circle):
                continue

            perm_anom = perm0.copy()
            perm_anom[in_circle] = perm0[in_circle] + float(anomaly_delta)

            v1 = np.asarray(fwd.solve_eit(perm=perm_anom), dtype=float).ravel()
            dv = v1 - v0

            jac_anom, _ = fwd.compute_jac(perm=perm_anom)
            jac_anom = np.asarray(jac_anom, dtype=float)
            sens_elem = np.sum(np.abs(jac_anom), axis=0)
            current_proxy = np.sum(np.abs(jac_anom), axis=1)

            signal_change_l2 = float(np.linalg.norm(dv) / (np.linalg.norm(v0) + eps))
            signal_change_linf = float(np.max(np.abs(dv)) / (np.max(np.abs(v0)) + eps))
            signal_change = signal_change_l2

            sensitivity_uniformity = _safe_uniformity(sens_elem)
            sensitivity_strength = float(np.mean(sens_elem[in_circle]) / (np.mean(sens_elem) + eps))
            current_distribution = _safe_uniformity(current_proxy)

            ds = jac_solver.solve(v1, v0, normalize=True)
            ds = np.real(np.asarray(ds, dtype=float).ravel())

            pred_mask = _prediction_mask_from_reconstruction(ds, n_true=np.count_nonzero(in_circle))
            recon_dice = _dice_score(in_circle, pred_mask)
            recon_center = _masked_centroid(centroids, np.abs(ds))
            center_error_norm = float(np.linalg.norm(recon_center - center) / domain_diag)
            reconstruction = float(recon_dice * (1.0 / (1.0 + center_error_norm)))

            composite = (
                float(weights["signal_change"]) * signal_change
                + float(weights["sensitivity_uniformity"]) * sensitivity_uniformity
                + float(weights["sensitivity_strength"]) * sensitivity_strength
                + float(weights["reconstruction"]) * reconstruction
                + float(weights["current_distribution"]) * current_distribution
            )

            cases.append(
                {
                    "radius": float(radius),
                    "center_x": float(center[0]),
                    "center_y": float(center[1]),
                    "n_anomaly_elements": int(np.count_nonzero(in_circle)),
                    "signal_change": float(signal_change),
                    "signal_change_l2": float(signal_change_l2),
                    "signal_change_linf": float(signal_change_linf),
                    "sensitivity_uniformity": float(sensitivity_uniformity),
                    "sensitivity_strength": float(sensitivity_strength),
                    "reconstruction": float(reconstruction),
                    "reconstruction_dice": float(recon_dice),
                    "reconstruction_center_error_norm": float(center_error_norm),
                    "current_distribution": float(current_distribution),
                    "composite_score": float(composite),
                }
            )

    if not cases:
        raise RuntimeError("No anomaly cases were generated for benchmarking.")

    cases_sorted = sorted(cases, key=lambda d: d["composite_score"], reverse=True)

    metrics = [
        "signal_change",
        "sensitivity_uniformity",
        "sensitivity_strength",
        "reconstruction",
        "current_distribution",
        "composite_score",
    ]
    summary = {
        "n_cases": len(cases_sorted),
        "best_case": cases_sorted[0],
        "metric_means": {m: float(np.mean([c[m] for c in cases_sorted])) for m in metrics},
        "metric_mins": {m: float(np.min([c[m] for c in cases_sorted])) for m in metrics},
        "metric_maxs": {m: float(np.max([c[m] for c in cases_sorted])) for m in metrics},
        "objective_weights": dict(weights),
    }

    return {
        "mesh_obj": mesh_obj,
        "cases": cases_sorted,
        "summary": summary,
    }


def save_benchmark_cases_csv(cases, output_csv):
    if not cases:
        return
    fieldnames = list(cases[0].keys())
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cases)


def _build_element_adjacency_from_triangles(triangles):
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


def load_saved_result(npz_path):
    with np.load(npz_path, allow_pickle=False) as data:
        required = {"best_perm", "best_sensitivity", "best_state"}
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"Saved result is missing required arrays: {', '.join(missing)}")

        out = {
            "best_perm": np.asarray(data["best_perm"], dtype=float).ravel(),
            "best_sensitivity": np.asarray(data["best_sensitivity"], dtype=float).ravel(),
            "best_state": np.asarray(data["best_state"], dtype=bool).ravel(),
            "node": np.asarray(data["node"], dtype=float) if "node" in data.files else None,
            "element": np.asarray(data["element"], dtype=int) if "element" in data.files else None,
            "el_pos": np.asarray(data["el_pos"], dtype=int).ravel() if "el_pos" in data.files else None,
        }
    return out


def benchmark_saved_result(npz_path, benchmark_name="uniformity", benchmark_weights=None):
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
    if payload.get("element") is not None:
        adjacency = _build_element_adjacency_from_triangles(payload["element"])

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
            raise ValueError("Touch benchmarks require node, element, and el_pos data in the saved NPZ.")

        from pyeit.eit.fem import EITForward
        import pyeit.eit.protocol as protocol
        from pyeit.mesh import PyEITMesh

        mesh_obj = PyEITMesh(
            node=np.asarray(payload["node"], dtype=float),
            element=np.asarray(payload["element"], dtype=int),
            el_pos=np.asarray(payload["el_pos"], dtype=int),
            perm=np.asarray(payload["best_perm"], dtype=float),
        )
        protocol_obj = protocol.create(
            n_el=int(np.asarray(mesh_obj.el_pos).size),
            dist_exc=max(1, int(np.asarray(mesh_obj.el_pos).size) // 2),
            step_meas=1,
            parser_meas="rotate_meas",
        )
        fwd = EITForward(mesh_obj, protocol_obj)
        jacobian, _ = fwd.compute_jac(perm=np.asarray(payload["best_perm"], dtype=float))
        jacobian = np.asarray(jacobian, dtype=float)

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


def print_benchmark_summary(summary, top_cases=None):
    print(f"Anomaly benchmark cases: {summary['n_cases']}")
    print("Objective weights:")
    for k, v in summary["objective_weights"].items():
        print(f"  - {k}: {float(v):.4f}")

    best = summary["best_case"]
    print("Best case (max composite_score):")
    print(
        "  "
        f"radius={best['radius']:.4f}, center=({best['center_x']:.4f}, {best['center_y']:.4f}), "
        f"signal_change={best['signal_change']:.6e}, composite={best['composite_score']:.6e}"
    )

    print("Metric means:")
    for m, val in summary["metric_means"].items():
        print(f"  - {m}: {val:.6e}")

    if top_cases:
        print(f"Top {len(top_cases)} cases:")
        for i, case in enumerate(top_cases, start=1):
            print(
                f"  {i:2d}. r={case['radius']:.4f} c=({case['center_x']:.3f},{case['center_y']:.3f}) "
                f"sig={case['signal_change']:.3e} recon={case['reconstruction']:.3e} "
                f"curr={case['current_distribution']:.3e} comp={case['composite_score']:.3e}"
            )


def plot_anomaly_benchmark_cases(cases, top_k=12):
    if not cases:
        return

    top_k = int(max(1, min(top_k, len(cases))))
    rows = cases[:top_k]

    radii = np.array([r["radius"] for r in rows], dtype=float)
    signal = np.array([r["signal_change"] for r in rows], dtype=float)
    sens_u = np.array([r["sensitivity_uniformity"] for r in rows], dtype=float)
    sens_s = np.array([r["sensitivity_strength"] for r in rows], dtype=float)
    recon = np.array([r["reconstruction"] for r in rows], dtype=float)
    curr = np.array([r["current_distribution"] for r in rows], dtype=float)
    comp = np.array([r["composite_score"] for r in rows], dtype=float)
    cx = np.array([r["center_x"] for r in rows], dtype=float)
    cy = np.array([r["center_y"] for r in rows], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

    ax0 = axes[0]
    im0 = ax0.scatter(cx, cy, c=comp, s=120 + 900 * radii, cmap="viridis", edgecolor="k", alpha=0.85)
    ax0.set_title("Top Cases: Position and Radius")
    ax0.set_xlabel("center_x")
    ax0.set_ylabel("center_y")
    ax0.set_aspect("equal")
    ax0.grid(True, alpha=0.25)
    fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04, label="composite_score")

    ax1 = axes[1]
    idx = np.arange(top_k)
    ax1.plot(idx, signal, "-o", label="signal_change", linewidth=1.5)
    ax1.plot(idx, sens_u, "-o", label="sens_uniformity", linewidth=1.2)
    ax1.plot(idx, sens_s, "-o", label="sens_strength", linewidth=1.2)
    ax1.plot(idx, recon, "-o", label="reconstruction", linewidth=1.2)
    ax1.plot(idx, curr, "-o", label="current_distribution", linewidth=1.2)
    ax1.set_title("Top Cases: Benchmark Metrics")
    ax1.set_xlabel("rank (0 = best)")
    ax1.set_ylabel("metric value")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")

    ax2 = axes[2]
    ax2.bar(idx, comp)
    ax2.set_title("Top Cases: Composite Score")
    ax2.set_xlabel("rank (0 = best)")
    ax2.set_ylabel("composite_score")
    ax2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_results(
    mesh_obj,
    p1,
    p2,
    best_perm,
    best_sensitivity,
    best_hist,
    curr_hist,
    best_score,
    best_theta=None,
    field_shape_cfg=None,
    comparison_rows=None,
):
    pts = mesh_obj.node
    tri = mesh_obj.element
    el_pos = mesh_obj.el_pos

    xs, ys = pts[:, 0], pts[:, 1]

    comparison_rows = comparison_rows or []
    rows = [
        {
            "name": "Best Optimized",
            "perm": best_perm,
            "sensitivity": best_sensitivity,
            "score": best_score,
        }
    ] + comparison_rows

    fig, axes = plt.subplots(len(rows), 3, figsize=(18, 5.2 * len(rows)))
    if len(rows) == 1:
        axes = np.asarray([axes])

    fig.suptitle("Global Binary Optimization (Benchmark Objective)", fontsize=13)

    for r, row in enumerate(rows):
        ax0 = axes[r, 0]
        ax1 = axes[r, 1]
        ax2 = axes[r, 2]

        perm_row = np.asarray(row["perm"]).ravel()
        sens_row = np.asarray(row["sensitivity"]).ravel()

        im0 = ax0.tripcolor(xs, ys, tri, perm_row, shading="flat", cmap="RdBu_r")
        if (
            r == 0
            and field_shape_cfg is not None
            and str(field_shape_cfg.get("model", "rbf")).strip().lower() in {"fractal", "branching"}
            and best_theta is not None
        ):
            try:
                from mesh_generation import evaluate_branching_grid, evaluate_fractal_grid
            except ImportError:
                from .mesh_generation import evaluate_branching_grid, evaluate_fractal_grid
            if str(field_shape_cfg.get("model", "rbf")).strip().lower() == "branching":
                Xf, Yf, Ff, Sf, _ = evaluate_branching_grid(
                    best_theta,
                    field_shape_cfg,
                    p1,
                    p2,
                    return_strength=True,
                )
            else:
                Xf, Yf, Ff, Sf, _ = evaluate_fractal_grid(
                    best_theta,
                    field_shape_cfg,
                    p1,
                    p2,
                    return_strength=True,
                )
            ax0.contourf(Xf, Yf, Sf, levels=18, cmap="magma", alpha=0.18)
            ax0.contour(Xf, Yf, Ff, levels=[0.0], colors=["#00ffff"], linewidths=1.3)

        ax0.plot(xs[el_pos], ys[el_pos], "ko", markersize=3)
        ax0.set_aspect("equal")
        ax0.set_xlim(p1[0] - 0.2, p2[0] + 0.2)
        ax0.set_ylim(p1[1] - 0.2, p2[1] + 0.2)
        ax0.set_title(f"{row['name']} Conductivity")
        ax0.set_xlabel("x")
        ax0.set_ylabel("y")
        fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04, label="Conductivity")

        im1 = ax1.tripcolor(xs, ys, tri, sens_row, shading="flat", cmap="viridis")
        ax1.set_aspect("equal")
        ax1.set_xlim(p1[0] - 0.2, p2[0] + 0.2)
        ax1.set_ylim(p1[1] - 0.2, p2[1] + 0.2)
        ax1.set_title(f"{row['name']} Sensitivity")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label="||J[:, elem]||")

        if r == 0:
            ax2.plot(best_hist, label="Best-so-far score", linewidth=1.8)
            ax2.plot(curr_hist, label="Current score", linewidth=1.0, alpha=0.7)
            ax2.set_title("Optimization Progress")
            ax2.set_xlabel("Iteration")
            ax2.set_ylabel("Benchmark Score")
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc="best")
        else:
            ax2.axis("off")
            ax2.text(
                0.03,
                0.85,
                f"{row['name']}\nScore: {row['score']:.6e}\n"
                f"Sensitivity min: {np.min(sens_row):.6e}\n"
                f"Sensitivity max: {np.max(sens_row):.6e}",
                va="top",
                ha="left",
                fontsize=10,
            )

    plt.tight_layout()
    plt.show()


def _demo_run(benchmark_name="uniformity"):
    rng = np.random.default_rng(7)
    synthetic_sensitivity = np.abs(rng.normal(loc=0.5, scale=0.3, size=120))
    synthetic_state = rng.random(120) > 0.45
    synthetic_adjacency = [
        [j for j in (i - 1, i + 1) if 0 <= j < 120]
        for i in range(120)
    ]

    fn = get_benchmark(benchmark_name)
    weights = get_profile_weights(benchmark_name if benchmark_name in BENCHMARK_PROFILES else "uniformity")
    score = fn(
        sensitivity=synthetic_sensitivity,
        state=synthetic_state,
        adjacency=synthetic_adjacency,
        weights=weights,
    )
    print(f"Benchmark '{benchmark_name}' demo score: {float(score):.6e}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark registry and scoring smoke test.")
    parser.add_argument(
        "--benchmark",
        default="uniformity",
        help="Benchmark name to test.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List registered benchmark names and exit.",
    )
    parser.add_argument(
        "--anomaly-benchmark",
        action="store_true",
        help="Run circular anomaly benchmark sweep (size/location) instead of synthetic demo.",
    )
    parser.add_argument("--n-el", type=int, default=16, help="Number of electrodes for anomaly benchmark.")
    parser.add_argument("--h0", type=float, default=0.08, help="Mesh target size for anomaly benchmark.")
    parser.add_argument("--radius-min", type=float, default=0.10, help="Minimum anomaly radius.")
    parser.add_argument("--radius-max", type=float, default=0.30, help="Maximum anomaly radius.")
    parser.add_argument("--radius-steps", type=int, default=4, help="Number of radius samples.")
    parser.add_argument(
        "--centers-per-radius",
        type=int,
        default=9,
        help="Number of anomaly centers sampled per radius.",
    )
    parser.add_argument(
        "--anomaly-delta",
        type=float,
        default=8.0,
        help="Absolute conductivity increment for anomaly elements.",
    )
    parser.add_argument(
        "--background-cond",
        type=float,
        default=1.0,
        help="Background conductivity.",
    )
    parser.add_argument(
        "--plot-top-k",
        type=int,
        default=12,
        help="Number of top benchmark cases to plot.",
    )
    parser.add_argument(
        "--save-csv",
        default=None,
        help="Optional output CSV path to save all anomaly benchmark rows.",
    )
    parser.add_argument(
        "--load-run",
        default=None,
        help="Path to saved optimization artifact NPZ to score (best conductivity/sensitivity/state).",
    )
    parser.add_argument(
        "--save-score-json",
        default=None,
        help="Optional JSON path to save score summary from --load-run.",
    )
    parser.add_argument("--score-entropy-weight", type=float, default=None, help="Override entropy weight for loaded-run scoring.")
    parser.add_argument("--score-isolated-weight", type=float, default=None, help="Override isolated penalty weight for loaded-run scoring.")
    parser.add_argument("--score-disconnected-weight", type=float, default=None, help="Override disconnected penalty weight for loaded-run scoring.")
    args = parser.parse_args()

    if args.list:
        print("Registered benchmarks:")
        for name in list_benchmarks():
            print(f"- {name}")
        return

    if args.anomaly_benchmark:
        radius_values = np.linspace(
            float(args.radius_min),
            float(args.radius_max),
            int(max(1, args.radius_steps)),
        )
        out = run_circular_anomaly_benchmark(
            n_el=args.n_el,
            h0=args.h0,
            radius_values=radius_values,
            centers_per_radius=args.centers_per_radius,
            anomaly_delta=args.anomaly_delta,
            background_cond=args.background_cond,
        )
        cases = out["cases"]
        summary = out["summary"]
        print_benchmark_summary(summary, top_cases=cases[: int(max(1, args.plot_top_k))])
        if args.save_csv:
            save_benchmark_cases_csv(cases, args.save_csv)
            print(f"Saved anomaly benchmark rows to: {args.save_csv}")
        plot_anomaly_benchmark_cases(cases, top_k=args.plot_top_k)
        return

    if args.load_run:
        overrides = {}
        if args.score_entropy_weight is not None:
            overrides["entropy"] = float(args.score_entropy_weight)
        if args.score_isolated_weight is not None:
            overrides["isolated_penalty"] = float(args.score_isolated_weight)
        if args.score_disconnected_weight is not None:
            overrides["disconnected_penalty"] = float(args.score_disconnected_weight)

        benchmark_weights = None
        if overrides:
            profile_key = args.benchmark if args.benchmark in BENCHMARK_PROFILES else "uniformity"
            benchmark_weights = get_profile_weights(profile_key, overrides=overrides)

        scored = benchmark_saved_result(
            npz_path=args.load_run,
            benchmark_name=args.benchmark,
            benchmark_weights=benchmark_weights,
        )

        print("Loaded saved run benchmark summary:")
        print(f"  source: {scored['source_npz']}")
        print(f"  benchmark: {scored['benchmark_name']}")
        print(f"  score: {scored['score']:.6e}")
        print(f"  elements: {scored['n_elements']} (high={scored['n_high_elements']})")
        print(
            "  sensitivity(min/mean/max): "
            f"{scored['sensitivity_min']:.6e} / {scored['sensitivity_mean']:.6e} / {scored['sensitivity_max']:.6e}"
        )
        if scored["high_elements_connected"] is not None:
            print(f"  high connected: {scored['high_elements_connected']}")
            print(f"  isolated high elements: {scored['isolated_high_elements']}")

        if args.save_score_json:
            with open(args.save_score_json, "w", encoding="utf-8") as f:
                json.dump(scored, f, indent=2)
            print(f"Saved loaded-run score summary to: {args.save_score_json}")
        return

    _demo_run(benchmark_name=args.benchmark)


if __name__ == "__main__":
    main()
