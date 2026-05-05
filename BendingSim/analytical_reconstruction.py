"""Analytical reconstruction benchmarks for EIT meshes.

Tests the ability to localize and reconstruct:
1. Touch points at varying distances from mesh boundary
2. Geometric shapes: cross, star, c-shape
"""

from __future__ import annotations

import numpy as np
from typing import Any


def touch_point_at_distance_from_edge(
    mesh_centroids: np.ndarray,
    bbox: tuple[float, float, float, float],
    distance_frac: float = 0.2,
) -> np.ndarray:
    """
    Generate a synthetic conductivity anomaly centered at a single point
    at a specified distance from the nearest mesh edge.
    
    Args:
        mesh_centroids: Array of shape (n_elements, 2) with element centers
        bbox: Tuple (x_min, x_max, y_min, y_max) defining mesh bounding box
        distance_frac: Fraction of mesh half-width for distance from edge (0.0 = edge, 1.0 = center)
    
    Returns:
        Synthetic conductivity state of shape (n_elements,)
    """
    x_min, x_max, y_min, y_max = bbox
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    w = x_max - x_min
    h = y_max - y_min
    
    # Distance from edge along shortest direction
    edge_dist = min(w, h) * 0.5 * max(0.0, distance_frac)
    
    # Touch point at center, inset from edge by edge_dist
    touch_x = cx
    touch_y = cy - (min(w, h) * 0.5 - edge_dist)
    
    # Find nearest element
    dist_sq = (mesh_centroids[:, 0] - touch_x) ** 2 + (mesh_centroids[:, 1] - touch_y) ** 2
    nearest_idx = np.argmin(dist_sq)
    
    state = np.zeros(mesh_centroids.shape[0], dtype=bool)
    state[nearest_idx] = True
    
    return state


def shape_cross(
    mesh_centroids: np.ndarray,
    bbox: tuple[float, float, float, float],
    thickness_frac: float = 0.15,
) -> np.ndarray:
    """
    Generate a cross-shaped conductivity anomaly.
    
    Args:
        mesh_centroids: Array of shape (n_elements, 2)
        bbox: Tuple (x_min, x_max, y_min, y_max)
        thickness_frac: Thickness of cross arms as fraction of mesh size
    
    Returns:
        Bool array marking cross elements
    """
    x_min, x_max, y_min, y_max = bbox
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    w = x_max - x_min
    h = y_max - y_min
    thickness = min(w, h) * thickness_frac
    
    state = np.zeros(mesh_centroids.shape[0], dtype=bool)
    
    # Horizontal bar
    h_mask = (np.abs(mesh_centroids[:, 1] - cy) < thickness / 2) & \
             (mesh_centroids[:, 0] >= x_min) & (mesh_centroids[:, 0] <= x_max)
    
    # Vertical bar
    v_mask = (np.abs(mesh_centroids[:, 0] - cx) < thickness / 2) & \
             (mesh_centroids[:, 1] >= y_min) & (mesh_centroids[:, 1] <= y_max)
    
    state = h_mask | v_mask
    return state


def shape_star(
    mesh_centroids: np.ndarray,
    bbox: tuple[float, float, float, float],
    n_arms: int = 5,
    radius_frac: float = 0.3,
    thickness_frac: float = 0.10,
) -> np.ndarray:
    """
    Generate a star-shaped conductivity anomaly.
    
    Args:
        mesh_centroids: Array of shape (n_elements, 2)
        bbox: Tuple (x_min, x_max, y_min, y_max)
        n_arms: Number of star arms
        radius_frac: Radius of star as fraction of mesh half-size
        thickness_frac: Arm thickness as fraction of mesh size
    
    Returns:
        Bool array marking star elements
    """
    x_min, x_max, y_min, y_max = bbox
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    min_size = min(x_max - x_min, y_max - y_min)
    radius = min_size * 0.5 * radius_frac
    thickness = min_size * thickness_frac
    
    state = np.zeros(mesh_centroids.shape[0], dtype=bool)
    
    for arm in range(n_arms):
        angle = 2.0 * np.pi * arm / n_arms
        arm_x = cx + radius * np.cos(angle)
        arm_y = cy + radius * np.sin(angle)
        
        # Draw line from center to arm end
        dx = arm_x - cx
        dy = arm_y - cy
        length = np.sqrt(dx * dx + dy * dy)
        if length > 0:
            dx /= length
            dy /= length
        
        # Find elements close to this line
        for i, (ex, ey) in enumerate(mesh_centroids):
            # Distance from point to line segment
            t = max(0.0, min(1.0, ((ex - cx) * dx + (ey - cy) * dy) / (length + 1e-10)))
            px = cx + t * dx * length
            py = cy + t * dy * length
            dist = np.sqrt((ex - px) ** 2 + (ey - py) ** 2)
            if dist < thickness / 2:
                state[i] = True
    
    return state


def shape_c(
    mesh_centroids: np.ndarray,
    bbox: tuple[float, float, float, float],
    radius_frac: float = 0.25,
    thickness_frac: float = 0.12,
    opening_angle: float = 90.0,
) -> np.ndarray:
    """
    Generate a C-shaped conductivity anomaly.
    
    Args:
        mesh_centroids: Array of shape (n_elements, 2)
        bbox: Tuple (x_min, x_max, y_min, y_max)
        radius_frac: Radius of C as fraction of mesh half-size
        thickness_frac: Thickness as fraction of mesh size
        opening_angle: Opening angle in degrees (default 90)
    
    Returns:
        Bool array marking C-shaped elements
    """
    x_min, x_max, y_min, y_max = bbox
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    min_size = min(x_max - x_min, y_max - y_min)
    radius = min_size * 0.5 * radius_frac
    thickness = min_size * thickness_frac
    opening_rad = opening_angle * np.pi / 180.0
    
    state = np.zeros(mesh_centroids.shape[0], dtype=bool)
    
    # C-shape: arc from -opening/2 to +opening/2
    start_angle = -opening_rad / 2.0
    end_angle = opening_rad / 2.0
    
    for i, (ex, ey) in enumerate(mesh_centroids):
        # Distance from element to center
        dx = ex - cx
        dy = ey - cy
        r = np.sqrt(dx * dx + dy * dy)
        angle = np.arctan2(dy, dx)
        
        # Check if within arc range
        angle_in_range = start_angle <= angle <= end_angle
        
        if angle_in_range and abs(r - radius) < thickness / 2:
            state[i] = True
    
    return state


def compute_reconstruction_quality(
    jacobian: np.ndarray,
    synthetic_state: np.ndarray,
    element_conductivity: np.ndarray,
) -> dict[str, float]:
    """
    Compute reconstruction quality metrics comparing synthetic conductivity
    to sensitivity-weighted reconstruction.
    
    Args:
        jacobian: Shape (n_measurements, n_elements)
        synthetic_state: Bool array marking synthetic anomaly
        element_conductivity: Current conductivity state
    
    Returns:
        Dict with quality metrics (correlation, l2_error, snr, etc.)
    """
    # Sensitivity-based reconstruction
    sensitivity = np.abs(jacobian).sum(axis=0)
    sensitivity = sensitivity / (np.max(sensitivity) + 1e-10)
    
    # Normalize states for comparison
    synthetic_norm = synthetic_state.astype(float) / (np.sum(synthetic_state) + 1e-10)
    sens_norm = sensitivity / (np.max(sensitivity) + 1e-10)
    
    # Correlation between synthetic and sensitivity
    correlation = np.corrcoef(synthetic_norm, sens_norm)[0, 1]
    if np.isnan(correlation):
        correlation = 0.0
    
    # L2 error
    l2_error = np.sqrt(np.mean((synthetic_norm - sens_norm) ** 2))
    
    # SNR: energy in synthetic region vs noise region
    synthetic_indices = np.where(synthetic_state)[0]
    other_indices = np.where(~synthetic_state)[0]
    
    if synthetic_indices.size > 0 and other_indices.size > 0:
        synthetic_energy = np.mean(sensitivity[synthetic_indices])
        noise_energy = np.mean(sensitivity[other_indices])
        snr = 10.0 * np.log10(max(synthetic_energy / (noise_energy + 1e-10), 1e-6))
    else:
        snr = 0.0
    
    # Localization precision (sum of sensitivity within synthetic region)
    localization = float(np.sum(sensitivity[synthetic_indices]) / (np.sum(sensitivity) + 1e-10))
    
    return {
        "correlation": float(correlation) if np.isfinite(correlation) else 0.0,
        "l2_error": float(l2_error),
        "snr_db": float(snr),
        "localization": float(localization),
    }


def run_all_reconstructions(
    jacobian: np.ndarray,
    mesh_centroids: np.ndarray,
    bbox: tuple[float, float, float, float],
    element_conductivity: np.ndarray,
) -> dict[str, Any]:
    """
    Run all analytical reconstruction tests on a mesh.
    
    Args:
        jacobian: Shape (n_measurements, n_elements)
        mesh_centroids: Shape (n_elements, 2)
        bbox: Mesh bounding box
        element_conductivity: Current element conductivity
    
    Returns:
        Dict with results for each test
    """
    results = {}
    
    # Touch point tests at different distances
    touch_distances = [0.2, 0.5, 0.8]
    touch_results = {}
    for dist in touch_distances:
        synthetic_state = touch_point_at_distance_from_edge(mesh_centroids, bbox, dist)
        quality = compute_reconstruction_quality(jacobian, synthetic_state, element_conductivity)
        touch_results[f"distance_{dist:.1f}"] = quality
    results["touch_points"] = touch_results
    
    # Shape tests
    shape_results = {}
    
    cross_state = shape_cross(mesh_centroids, bbox)
    shape_results["cross"] = compute_reconstruction_quality(jacobian, cross_state, element_conductivity)
    
    star_state = shape_star(mesh_centroids, bbox)
    shape_results["star"] = compute_reconstruction_quality(jacobian, star_state, element_conductivity)
    
    c_state = shape_c(mesh_centroids, bbox)
    shape_results["c_shape"] = compute_reconstruction_quality(jacobian, c_state, element_conductivity)
    
    results["shapes"] = shape_results
    
    return results
