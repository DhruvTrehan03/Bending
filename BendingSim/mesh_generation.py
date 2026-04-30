"""Standalone mesh generation and field-design helpers."""

from __future__ import annotations

import argparse
import numpy as np
import matplotlib.pyplot as plt
import pyeit.eit.protocol as protocol
import pyeit.mesh as mesh
from pyeit.mesh.shape import rectangle


def _clean_cfg_string(value, default=""):
    if value is None:
        return str(default).strip().lower()
    text = str(value)
    for sep in ("#", ";"):
        if sep in text:
            text = text.split(sep, 1)[0]
    return text.strip().lower()


def rect_electrode_positions(n_el, p1, p2):
    """Uniformly place electrodes along the full rectangle boundary."""
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


def build_model(n_el=16, h0=0.1, p1=None, p2=None):
    """Build a rectangle model with all elements optimizable."""
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

    tri = mesh_obj.element
    variable_mask = np.ones(tri.shape[0], dtype=bool)
    return mesh_obj, protocol_obj, variable_mask, p1, p2


def visualize_mesh(mesh_obj, p1, p2, title="Rectangle Mesh", show_element_ids=False):
    """Visualize generated triangular mesh and electrode locations."""
    pts = mesh_obj.node
    tri = mesh_obj.element
    el_pos = mesh_obj.el_pos

    xs = pts[:, 0]
    ys = pts[:, 1]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.triplot(xs, ys, tri, color="#4f5d75", linewidth=0.6, alpha=0.8)
    ax.scatter(xs[el_pos], ys[el_pos], c="#d62828", s=28, zorder=3, label="Electrodes")

    if show_element_ids:
        centroids = np.mean(pts[tri], axis=1)
        for idx, (cx, cy, _cz) in enumerate(centroids):
            ax.text(cx, cy, str(idx), fontsize=6, color="#1d3557", alpha=0.75)

    ax.set_aspect("equal")
    ax.set_xlim(p1[0] - 0.2, p2[0] + 0.2)
    ax.set_ylim(p1[1] - 0.2, p2[1] + 0.2)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def _read_mesh_generation_cfg(config_path):
    import configparser

    cfg = configparser.ConfigParser()
    cfg.read(config_path)

    def _getint(section, option, fallback):
        if cfg.has_option(section, option):
            return cfg.getint(section, option)
        return fallback

    def _getfloat(section, option, fallback):
        if cfg.has_option(section, option):
            return cfg.getfloat(section, option)
        return fallback

    def _getbool(section, option, fallback):
        if cfg.has_option(section, option):
            return cfg.getboolean(section, option)
        return fallback

    def _getstr(section, option, fallback):
        if cfg.has_option(section, option):
            return _clean_cfg_string(cfg.get(section, option), default=fallback)
        return fallback

    # Preferred section: [mesh_generation]. Optional fallback: [combined].
    section = "mesh_generation"
    return {
        "n_el": _getint(section, "n_el", _getint("combined", "n_el", 16)),
        "h0": _getfloat(section, "h0", _getfloat("combined", "h0", 0.1)),
        "model": _getstr(section, "model", _getstr("combined", "field_model", "rbf")),
        "rbf_rows": _getint(section, "rbf_rows", 4),
        "rbf_cols": _getint(section, "rbf_cols", 5),
        "rbf_sigma_frac": _getfloat(section, "rbf_sigma_frac", 0.9),
        "rbf_weight_scale": _getfloat(section, "rbf_weight_scale", 0.35),
        "fractal_iter_max": _getint(section, "fractal_iter_max", 40),
        "fractal_type": _getstr(section, "fractal_type", "branching"),
        "fractal_power": _getint(section, "fractal_power", 3),
        "fractal_shift_frac": _getfloat(section, "fractal_shift_frac", 0.55),
        "fractal_center_origin": _getbool(section, "fractal_center_origin", True),
        "fractal_threshold_init": _getfloat(section, "fractal_threshold_init", 0.0),
        "branching_depth_max": _getint(section, "branching_depth_max", 5),
        "branching_max_children": _getint(section, "branching_max_children", 2),
        "branching_child_angle_frac": _getfloat(section, "branching_child_angle_frac", 0.34),
        "branching_size_min_frac": _getfloat(section, "branching_size_min_frac", 0.55),
        "branching_size_max_frac": _getfloat(section, "branching_size_max_frac", 1.0),
        "show_mesh": _getbool(section, "show_mesh", True),
        "show_element_ids": _getbool(section, "show_element_ids", False),
    }


def build_element_adjacency(mesh_obj):
    """Build adjacency list for elements based on shared edges."""
    tri = mesh_obj.element
    n_elem = tri.shape[0]
    adjacency = [[] for _ in range(n_elem)]

    for i in range(n_elem):
        vertices_i = set(tri[i])
        for j in range(i + 1, n_elem):
            vertices_j = set(tri[j])
            if len(vertices_i & vertices_j) >= 2:
                adjacency[i].append(j)
                adjacency[j].append(i)

    return adjacency


def build_variable_element_geometry(mesh_obj, variable_mask, p1, p2):
    """Precompute geometry used by field parameterization."""
    tri = mesh_obj.element
    pts = mesh_obj.node
    centroids = np.mean(pts[tri], axis=1)[:, :2]

    variable_indices = np.where(variable_mask)[0]
    variable_centroids = centroids[variable_indices]

    x1, y1 = p1
    x2, y2 = p2
    w = x2 - x1
    h = y2 - y1
    perimeter = 2.0 * (w + h)

    x = variable_centroids[:, 0]
    y = variable_centroids[:, 1]

    dist_to_sides = np.column_stack([
        y - y1,
        x2 - x,
        y2 - y,
        x - x1,
    ])
    nearest_side = np.argmin(dist_to_sides, axis=1)
    inward_distance = np.take_along_axis(
        dist_to_sides,
        nearest_side[:, None],
        axis=1,
    ).ravel()

    boundary_s = np.empty(variable_centroids.shape[0], dtype=float)
    mask_bottom = nearest_side == 0
    boundary_s[mask_bottom] = x[mask_bottom] - x1

    mask_right = nearest_side == 1
    boundary_s[mask_right] = w + (y[mask_right] - y1)

    mask_top = nearest_side == 2
    boundary_s[mask_top] = w + h + (x2 - x[mask_top])

    mask_left = nearest_side == 3
    boundary_s[mask_left] = 2.0 * w + h + (y2 - y[mask_left])

    boundary_u = boundary_s / perimeter

    return {
        "variable_indices": variable_indices,
        "centroids": variable_centroids,
        "boundary_u": boundary_u,
        "inward_distance": inward_distance,
        "width": w,
        "height": h,
    }


def build_electrode_support_elements(mesh_obj, variable_mask, k_nearest=4):
    """Map each electrode to nearby variable elements."""
    tri = mesh_obj.element
    pts = mesh_obj.node
    centroids = np.mean(pts[tri], axis=1)[:, :2]
    variable_indices = np.where(variable_mask)[0]
    variable_centroids = centroids[variable_indices]

    electrode_points = pts[mesh_obj.el_pos][:, :2]
    supports = []
    for ept in electrode_points:
        d2 = np.sum((variable_centroids - ept) ** 2, axis=1)
        nearest = np.argsort(d2)[: max(1, int(k_nearest))]
        supports.append(nearest.astype(int))
    return supports


def parameter_vector_size(rbf_rows, rbf_cols):
    return 1 + int(rbf_rows) * int(rbf_cols)


def branching_parameter_vector_size(field_shape_cfg):
    max_children = int(max(1, field_shape_cfg.get("branching_max_children", 2)))
    depth_max = int(max(0, field_shape_cfg.get("branching_depth_max", 5)))

    node_count = 0
    nodes_this_level = 1
    for _depth in range(depth_max + 1):
        node_count += nodes_this_level
        nodes_this_level *= max_children

    return 3 * node_count


def parameter_vector_size_from_cfg(field_shape_cfg, field_geometry=None):
    model = str(field_shape_cfg.get("model", "rbf")).strip().lower()
    if model == "rbf":
        return parameter_vector_size(
            int(field_shape_cfg["rbf_rows"]),
            int(field_shape_cfg["rbf_cols"]),
        )
    if model in {"branching", "fungal"}:
        return branching_parameter_vector_size(field_shape_cfg)
    if model == "fractal":
        center_locked = bool(field_shape_cfg.get("fractal_center_origin", True))
        return 5 if center_locked else 7
    if model == "element":
        # Element model: size determined from field_geometry
        if field_geometry is not None and "n_variable_elements" in field_geometry:
            return int(field_geometry["n_variable_elements"])
        # Fallback: return 0 if geometry not available yet (will be set later)
        return 0
    raise ValueError(f"Unsupported field-shape model: {model}")


def _fractal_type_from_cfg(field_shape_cfg):
    fractal_type = _clean_cfg_string(field_shape_cfg.get("fractal_type", "branching"), default="branching")
    allowed = {"branching"}
    if fractal_type not in allowed:
        raise ValueError(
            f"Unsupported field_shape.fractal_type '{fractal_type}'. "
            "Use 'branching'."
        )
    return fractal_type


def _wrap_angle(angle):
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def _blend_angles(angle_a, angle_b, blend):
    blend = float(np.clip(blend, 0.0, 1.0))
    delta = _wrap_angle(angle_b - angle_a)
    return _wrap_angle(angle_a + blend * delta)


def _nearest_electrode_angle(x, y, electrode_points):
    if electrode_points is None:
        return None

    electrode_points = np.asarray(electrode_points, dtype=float)
    if electrode_points.size == 0:
        return None

    deltas = electrode_points - np.array([x, y], dtype=float)
    nearest_idx = int(np.argmin(np.sum(deltas * deltas, axis=1)))
    dx, dy = deltas[nearest_idx]
    return float(np.arctan2(dy, dx))


def _electrode_array(electrode_points):
    if electrode_points is None:
        return np.empty((0, 2), dtype=float)
    arr = np.asarray(electrode_points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        return np.empty((0, 2), dtype=float)
    return arr


def _branching_segment_distance(xn, yn, x0, y0, x1, y1):
    vx = x1 - x0
    vy = y1 - y0
    denom = vx * vx + vy * vy + np.finfo(float).tiny
    wx = xn - x0
    wy = yn - y0
    t = np.clip((wx * vx + wy * vy) / denom, 0.0, 1.0)
    px = x0 + t * vx
    py = y0 + t * vy
    return np.sqrt((xn - px) ** 2 + (yn - py) ** 2)


def _deterministic_signed_noise(*values):
    seed = 0.0
    coeff = 0.7548776662466927
    for val in values:
        seed = seed * 1.618033988749895 + coeff * float(val)
        coeff = coeff * 1.4142135623730951 + 0.2718281828459045
    raw = np.sin(seed * 12.9898 + 78.233) * 43758.5453123
    frac = raw - np.floor(raw)
    return float(2.0 * frac - 1.0)


def _branching_fractal_params_from_cfg(field_shape_cfg, theta):
    theta = np.asarray(theta, dtype=float).ravel()
    threshold = 0.5 * (float(np.clip(theta[0], -1.0, 1.0)) + 1.0)
    c_real = float(np.clip(theta[1], -1.0, 1.0))
    c_imag = float(np.clip(theta[2], -1.0, 1.0))
    zoom = float(2.0 ** (2.0 * float(np.clip(theta[3], -1.0, 1.0))))
    center_locked = bool(field_shape_cfg.get("fractal_center_origin", True))

    if center_locked:
        shift_x = 0.0
        shift_y = 0.0
        angle = np.pi * float(np.clip(theta[4], -1.0, 1.0))
    else:
        shift_frac = float(field_shape_cfg["fractal_shift_frac"])
        shift_x = shift_frac * float(np.clip(theta[4], -1.0, 1.0))
        shift_y = shift_frac * float(np.clip(theta[5], -1.0, 1.0))
        angle = np.pi * float(np.clip(theta[6], -1.0, 1.0))

    max_depth = int(max(1, field_shape_cfg.get("branching_depth_max", 6)))
    base_angle_frac = float(field_shape_cfg.get("branching_angle_frac", 0.34))
    length_decay = float(field_shape_cfg.get("branching_length_decay", 0.68))
    width_frac = float(field_shape_cfg.get("branching_width_frac", 0.12))
    width_decay = float(field_shape_cfg.get("branching_width_decay", 0.80))
    mirror_vertical = bool(field_shape_cfg.get("branching_mirror_vertical", False))
    aim_electrodes = bool(field_shape_cfg.get("branching_aim_electrodes", False))
    target_blend = float(field_shape_cfg.get("branching_target_blend", 0.0))
    root_x = float(field_shape_cfg.get("branching_root_x_frac", 0.18))
    root_y = float(field_shape_cfg.get("branching_root_y_frac", 0.0))
    force_touch_all = bool(field_shape_cfg.get("branching_force_touch_all_electrodes", True))
    seed_all_electrodes = bool(field_shape_cfg.get("branching_seed_all_electrodes", True))
    force_meet_center = bool(field_shape_cfg.get("branching_force_meet_center", True))
    meet_x = float(field_shape_cfg.get("branching_meet_x_frac", 0.0))
    meet_y = float(field_shape_cfg.get("branching_meet_y_frac", 0.0))
    meet_blend = float(field_shape_cfg.get("branching_meet_blend", 1.0))
    random_angle_frac = float(field_shape_cfg.get("branching_random_angle_frac", 0.0))
    random_center_boost = float(field_shape_cfg.get("branching_random_center_boost", 0.0))
    random_center_power = float(field_shape_cfg.get("branching_random_center_power", 1.0))
    electrode_points_all = field_shape_cfg.get("branching_electrode_points_norm")
    electrode_points = field_shape_cfg.get("branching_electrode_points_norm_right")
    if electrode_points is None:
        electrode_points = electrode_points_all

    branch_angle = np.pi * np.clip(base_angle_frac * (0.85 + 0.15 * c_real), 0.12, 0.50)
    branch_angle_left = branch_angle * (1.0 + 0.20 * c_imag)
    branch_angle_right = branch_angle * (1.0 - 0.20 * c_imag)
    root_length = 1.10 / max(0.65, zoom)
    root_width = max(0.03, width_frac / np.sqrt(max(1.0, zoom)))

    return {
        "threshold": threshold,
        "shift_x": shift_x,
        "shift_y": shift_y,
        "angle": angle,
        "max_depth": max_depth,
        "length_decay": float(np.clip(length_decay, 0.45, 0.92)),
        "width_decay": float(np.clip(width_decay, 0.55, 0.98)),
        "mirror_vertical": mirror_vertical,
        "aim_electrodes": aim_electrodes,
        "target_blend": target_blend,
        "root_x": root_x,
        "root_y": root_y,
        "force_touch_all": force_touch_all,
        "seed_all_electrodes": seed_all_electrodes,
        "force_meet_center": force_meet_center,
        "meet_x": meet_x,
        "meet_y": meet_y,
        "meet_blend": meet_blend,
        "random_angle": float(np.clip(np.pi * random_angle_frac, 0.0, 0.45 * np.pi)),
        "random_center_boost": float(max(0.0, random_center_boost)),
        "random_center_power": float(np.clip(random_center_power, 0.25, 5.0)),
        "noise_seed_real": c_real,
        "noise_seed_imag": c_imag,
        "electrode_points_all": electrode_points_all,
        "electrode_points": electrode_points,
        "branch_angle_left": float(branch_angle_left),
        "branch_angle_right": float(branch_angle_right),
        "root_length": float(root_length),
        "root_width": float(root_width),
    }


def _evaluate_branching_fractal_field_from_normalized_coords(xn, yn, theta, field_shape_cfg, return_strength=False):
    params = _branching_fractal_params_from_cfg(field_shape_cfg, theta)

    xn = np.asarray(xn, dtype=float) - params["shift_x"]
    yn = np.asarray(yn, dtype=float) - params["shift_y"]
    electrode_points = _electrode_array(params["electrode_points"])
    electrode_points_all = _electrode_array(params.get("electrode_points_all"))

    segments = []
    center_target = np.array([params["meet_x"], params["meet_y"]], dtype=float)
    root_radius = np.hypot(params["root_x"] - center_target[0], params["root_y"] - center_target[1])
    root_radius = max(root_radius, 1e-6)

    def random_angle_jitter(x_pos, y_pos, depth, branch_tag):
        if params["random_angle"] <= 0.0:
            return 0.0

        dist_to_center = np.hypot(x_pos - center_target[0], y_pos - center_target[1])
        center_proximity = 1.0 - np.clip(dist_to_center / root_radius, 0.0, 1.0)
        center_gain = 1.0 + params["random_center_boost"] * (center_proximity ** params["random_center_power"])
        signed_noise = _deterministic_signed_noise(
            x_pos,
            y_pos,
            depth,
            branch_tag,
            params["noise_seed_real"],
            params["noise_seed_imag"],
        )
        return params["random_angle"] * center_gain * signed_noise

    def guidance_angle(x_pos, y_pos, fallback_angle):
        if params["seed_all_electrodes"]:
            to_center = center_target - np.array([x_pos, y_pos], dtype=float)
            center_angle = float(np.arctan2(to_center[1], to_center[0]))
            return _blend_angles(fallback_angle, center_angle, params["meet_blend"])

        if not params["aim_electrodes"]:
            return float(fallback_angle)

        target_angle = _nearest_electrode_angle(x_pos, y_pos, electrode_points)
        if target_angle is None:
            return float(fallback_angle)

        return _blend_angles(fallback_angle, target_angle, params["target_blend"])

    def grow(x0, y0, length, angle, depth, branch_tag):
        if depth >= params["max_depth"] or length <= 1e-3:
            return

        x1 = x0 + length * np.cos(angle)
        y1 = y0 + length * np.sin(angle)
        segments.append((x0, y0, x1, y1, depth))

        next_length = length * params["length_decay"]
        left_base = guidance_angle(x1, y1, angle + params["branch_angle_left"])
        right_base = guidance_angle(x1, y1, angle - params["branch_angle_right"])
        left_angle = left_base + random_angle_jitter(x1, y1, depth + 1, branch_tag * 2.0 + 1.0)
        right_angle = right_base + random_angle_jitter(x1, y1, depth + 1, branch_tag * 2.0 + 2.0)
        grow(x1, y1, next_length, left_angle, depth + 1, branch_tag * 2.0 + 1.0)
        grow(x1, y1, next_length, right_angle, depth + 1, branch_tag * 2.0 + 2.0)

    if params["seed_all_electrodes"] and electrode_points_all.size > 0:
        for ex, ey in electrode_points_all:
            center_angle = float(np.arctan2(params["meet_y"] - ey, params["meet_x"] - ex))
            root_angle = _blend_angles(params["angle"], center_angle, params["meet_blend"])
            grow(float(ex), float(ey), params["root_length"], root_angle, 0, 1.0)
    else:
        root_angle = params["angle"]
        if params["aim_electrodes"]:
            root_target_angle = _nearest_electrode_angle(params["root_x"], params["root_y"], electrode_points)
            if root_target_angle is not None:
                root_angle = _blend_angles(root_angle, root_target_angle, params["target_blend"])
        grow(params["root_x"], params["root_y"], params["root_length"], root_angle, 0, 1.0)

    if params["seed_all_electrodes"] and params["force_meet_center"] and electrode_points_all.size > 0:
        cx = float(params["meet_x"])
        cy = float(params["meet_y"])
        for ex, ey in electrode_points_all:
            segments.append((float(ex), float(ey), cx, cy, 0))

    touch_points = electrode_points_all if electrode_points_all.size > 0 else electrode_points
    if params["force_touch_all"] and touch_points.size > 0:
        attachment_points = [(params["root_x"], params["root_y"])]
        attachment_points.extend((x1, y1) for _, _, x1, y1, _ in segments)
        attach = np.asarray(attachment_points, dtype=float)

        for ex, ey in touch_points:
            d2 = np.sum((attach - np.array([ex, ey], dtype=float)) ** 2, axis=1)
            idx = int(np.argmin(d2))
            x0, y0 = attach[idx]
            segments.append((x0, y0, float(ex), float(ey), 0))
            attach = np.vstack([attach, [float(ex), float(ey)]])

    if params["mirror_vertical"] and not params["seed_all_electrodes"]:
        mirrored_segments = [(-x0, y0, -x1, y1, depth) for x0, y0, x1, y1, depth in segments]
        segments = segments + mirrored_segments

    if not segments:
        field = -params["threshold"] * np.ones_like(xn, dtype=float)
        if return_strength:
            return field, np.zeros_like(xn, dtype=float), params["threshold"]
        return field

    strength = np.zeros_like(xn, dtype=float)
    for x0, y0, x1, y1, depth in segments:
        dist = _branching_segment_distance(xn, yn, x0, y0, x1, y1)
        width = params["root_width"] * (params["width_decay"] ** depth)
        width = max(width, 0.015)
        depth_weight = params["length_decay"] ** depth
        contribution = depth_weight * np.exp(-0.5 * (dist / width) ** 2)
        strength = np.maximum(strength, contribution)

    if np.max(strength) > 0.0:
        strength = strength / np.max(strength)

    field = strength - params["threshold"]
    if return_strength:
        return field, strength, params["threshold"]
    return field


def _evaluate_fractal_field_from_normalized_coords(xn, yn, theta, field_shape_cfg, return_strength=False):
    fractal_type = _fractal_type_from_cfg(field_shape_cfg)
    if fractal_type != "branching":
        raise ValueError(
            f"Unsupported field_shape.fractal_type '{fractal_type}'. Use 'branching'."
        )

    return _evaluate_branching_fractal_field_from_normalized_coords(
        xn=xn,
        yn=yn,
        theta=theta,
        field_shape_cfg=field_shape_cfg,
        return_strength=return_strength,
    )


def build_rbf_field_geometry(mesh_obj, variable_mask, p1, p2, rbf_rows, rbf_cols, sigma_frac):
    tri = mesh_obj.element
    pts = mesh_obj.node
    centroids = np.mean(pts[tri], axis=1)[:, :2]

    variable_indices = np.where(variable_mask)[0]
    variable_centroids = centroids[variable_indices]

    x1, y1 = p1
    x2, y2 = p2
    width = x2 - x1
    height = y2 - y1

    rbf_rows = int(max(1, rbf_rows))
    rbf_cols = int(max(1, rbf_cols))

    xs = np.linspace(x1 + 0.10 * width, x2 - 0.10 * width, rbf_cols)
    ys = np.linspace(y1 + 0.10 * height, y2 - 0.10 * height, rbf_rows)
    center_grid = np.array([(x, y) for y in ys for x in xs], dtype=float)

    dx = width / max(1, rbf_cols - 1)
    dy = height / max(1, rbf_rows - 1)
    sigma = float(max(1e-9, sigma_frac * min(dx, dy)))
    sigma2 = 2.0 * sigma * sigma

    deltas = variable_centroids[:, None, :] - center_grid[None, :, :]
    dist2 = np.sum(deltas * deltas, axis=2)
    basis = np.exp(-dist2 / sigma2)

    return {
        "variable_indices": variable_indices,
        "centroids": variable_centroids,
        "centers": center_grid,
        "basis": basis,
        "sigma": sigma,
        "width": width,
        "height": height,
    }


def build_fractal_field_geometry(mesh_obj, variable_mask, p1, p2):
    tri = mesh_obj.element
    pts = mesh_obj.node
    centroids = np.mean(pts[tri], axis=1)[:, :2]

    variable_indices = np.where(variable_mask)[0]
    variable_centroids = centroids[variable_indices]

    x1, y1 = p1
    x2, y2 = p2
    width = x2 - x1
    height = y2 - y1

    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    norm_scale = 0.5 * min(width, height)
    x_norm = (variable_centroids[:, 0] - cx) / norm_scale
    y_norm = (variable_centroids[:, 1] - cy) / norm_scale

    electrode_points = pts[mesh_obj.el_pos][:, :2]
    electrode_points_norm = (electrode_points - np.array([cx, cy], dtype=float)) / norm_scale
    electrode_points_norm_right = electrode_points_norm[electrode_points_norm[:, 0] >= 0.0]

    return {
        "variable_indices": variable_indices,
        "centroids": variable_centroids,
        "x_norm": x_norm,
        "y_norm": y_norm,
        "norm_scale": norm_scale,
        "width": width,
        "height": height,
        "electrode_points_norm": electrode_points_norm,
        "electrode_points_norm_right": electrode_points_norm_right,
    }


def build_branching_field_geometry(mesh_obj, variable_mask, p1, p2):
    geometry = build_fractal_field_geometry(mesh_obj, variable_mask, p1, p2)
    geometry["branching_seed_points_norm"] = geometry["electrode_points_norm"]
    geometry["branching_seed_points_norm_right"] = geometry["electrode_points_norm_right"]
    return geometry


def build_field_geometry(mesh_obj, variable_mask, p1, p2, field_shape_cfg):
    model = str(field_shape_cfg.get("model", "rbf")).strip().lower()
    if model == "rbf":
        return build_rbf_field_geometry(
            mesh_obj=mesh_obj,
            variable_mask=variable_mask,
            p1=p1,
            p2=p2,
            rbf_rows=field_shape_cfg["rbf_rows"],
            rbf_cols=field_shape_cfg["rbf_cols"],
            sigma_frac=field_shape_cfg["rbf_sigma_frac"],
        )
    if model == "fractal":
        return build_fractal_field_geometry(
            mesh_obj=mesh_obj,
            variable_mask=variable_mask,
            p1=p1,
            p2=p2,
        )
    if model in {"branching", "fungal"}:
        return build_branching_field_geometry(
            mesh_obj=mesh_obj,
            variable_mask=variable_mask,
            p1=p1,
            p2=p2,
        )
    if model == "element":
        # Element model: one parameter per variable element
        n_variable_elements = int(np.sum(variable_mask))
        tri = mesh_obj.element
        pts = mesh_obj.node
        centroids = np.mean(pts[tri], axis=1)[:, :2]
        return {
            "n_variable_elements": n_variable_elements,
            "centroids": centroids,
        }
    raise ValueError(f"Unsupported field-shape model: {model}")


def parameterized_state_from_theta(theta, field_geometry, field_shape_cfg):
    model = str(field_shape_cfg.get("model", "rbf")).strip().lower()
    theta = np.asarray(theta, dtype=float).ravel()
    expected = parameter_vector_size_from_cfg(field_shape_cfg, field_geometry)
    if theta.size != expected:
        raise ValueError(f"Expected {expected} parameters, got {theta.size}")

    if model == "rbf":
        bias = float(np.clip(theta[0], -1.0, 1.0))
        weights = np.clip(theta[1:], -1.0, 1.0)
        weight_scale = float(field_shape_cfg["rbf_weight_scale"])
        field = bias + weight_scale * field_geometry["basis"].dot(weights)
        state_high = field >= 0.0
        return np.asarray(state_high, dtype=bool), field

    if model == "fractal":
        field = _evaluate_fractal_field_from_normalized_coords(
            xn=field_geometry["x_norm"],
            yn=field_geometry["y_norm"],
            theta=theta,
            field_shape_cfg=field_shape_cfg,
        )
        state_high = field >= 0.0
        return np.asarray(state_high, dtype=bool), field

    if model in {"branching", "fungal"}:
        field = _evaluate_branching_tree_field_from_normalized_coords(
            xn=field_geometry["x_norm"],
            yn=field_geometry["y_norm"],
            theta=theta,
            field_shape_cfg=field_shape_cfg,
        )
        state_high = field >= 0.0
        return np.asarray(state_high, dtype=bool), field

    if model == "element":
        # Element model: each parameter directly determines if element is high (param > 0) or low (param <= 0)
        # Parameters are indexed by variable element indices
        params_clipped = np.clip(theta, -1.0, 1.0)
        state_high_var = params_clipped >= 0.0
        
        # Return the variable-element indexed state
        # The calling code (make_permittivity) will handle expansion to full mesh via variable_mask
        return np.asarray(state_high_var, dtype=bool), params_clipped

    raise ValueError(f"Unsupported field-shape model: {model}")


def evaluate_fractal_grid(theta, field_shape_cfg, p1, p2, grid_n=360, return_strength=False):
    theta = np.asarray(theta, dtype=float).ravel()
    expected = parameter_vector_size_from_cfg(field_shape_cfg)
    if theta.size != expected:
        raise ValueError(f"Expected {expected} parameters, got {theta.size}")

    x1, y1 = p1
    x2, y2 = p2
    width = x2 - x1
    height = y2 - y1
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    norm_scale = 0.5 * min(width, height)

    xs = np.linspace(x1, x2, int(max(64, grid_n)))
    ys = np.linspace(y1, y2, int(max(64, grid_n)))
    X, Y = np.meshgrid(xs, ys)

    xn = (X - cx) / norm_scale
    yn = (Y - cy) / norm_scale

    fractal_eval = _evaluate_fractal_field_from_normalized_coords(
        xn=xn,
        yn=yn,
        theta=theta,
        field_shape_cfg=field_shape_cfg,
        return_strength=return_strength,
    )
    if return_strength:
        field_grid, strength_grid, threshold = fractal_eval
        return X, Y, field_grid, strength_grid, threshold
    return X, Y, fractal_eval


def evaluate_branching_grid(theta, field_shape_cfg, p1, p2, grid_n=360, return_strength=False):
    theta = np.asarray(theta, dtype=float).ravel()
    expected = parameter_vector_size_from_cfg(field_shape_cfg)
    if theta.size != expected:
        raise ValueError(f"Expected {expected} parameters, got {theta.size}")

    x1, y1 = p1
    x2, y2 = p2
    xs = np.linspace(x1, x2, int(max(64, grid_n)))
    ys = np.linspace(y1, y2, int(max(64, grid_n)))
    X, Y = np.meshgrid(xs, ys)

    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    norm_scale = 0.5 * min(x2 - x1, y2 - y1)

    xn = (X - cx) / norm_scale
    yn = (Y - cy) / norm_scale
    branching_eval = _evaluate_branching_tree_field_from_normalized_coords(
        xn=xn,
        yn=yn,
        theta=theta,
        field_shape_cfg=field_shape_cfg,
        return_strength=return_strength,
    )
    if return_strength:
        field_grid, strength_grid, threshold = branching_eval
        return X, Y, field_grid, strength_grid, threshold
    return X, Y, branching_eval


def _branching_tree_node_count(field_shape_cfg):
    max_children = int(max(1, field_shape_cfg.get("branching_max_children", 2)))
    depth_max = int(max(0, field_shape_cfg.get("branching_depth_max", 5)))

    node_count = 0
    nodes_this_level = 1
    for _depth in range(depth_max + 1):
        node_count += nodes_this_level
        nodes_this_level *= max_children

    return node_count


def _branching_tree_child_index(node_index, child_slot, max_children):
    return node_index * max_children + child_slot + 1


def _branching_tree_params_from_cfg(field_shape_cfg, theta):
    theta = np.asarray(theta, dtype=float).ravel()
    node_count = _branching_tree_node_count(field_shape_cfg)
    expected = 3 * node_count
    if theta.size != expected:
        raise ValueError(f"Expected {expected} parameters, got {theta.size}")

    node_params = np.clip(theta.reshape(node_count, 3), -1.0, 1.0)
    max_children = int(max(1, field_shape_cfg.get("branching_max_children", 2)))
    depth_max = int(max(0, field_shape_cfg.get("branching_depth_max", 5)))
    angle_frac = float(field_shape_cfg.get("branching_angle_frac", 0.34))
    child_angle_frac = float(field_shape_cfg.get("branching_child_angle_frac", angle_frac))
    size_min_frac = float(field_shape_cfg.get("branching_size_min_frac", 0.55))
    size_max_frac = float(field_shape_cfg.get("branching_size_max_frac", 1.0))
    root_length_frac = float(field_shape_cfg.get("branching_root_length_frac", 1.05))
    length_decay = float(field_shape_cfg.get("branching_length_decay", 0.68))
    width_frac = float(field_shape_cfg.get("branching_width_frac", 0.12))
    width_decay = float(field_shape_cfg.get("branching_width_decay", 0.80))
    seed_all_electrodes = bool(field_shape_cfg.get("branching_seed_all_electrodes", True))
    center_x = float(field_shape_cfg.get("branching_meet_x_frac", 0.0))
    center_y = float(field_shape_cfg.get("branching_meet_y_frac", 0.0))
    root_x = float(field_shape_cfg.get("branching_root_x_frac", 0.18))
    root_y = float(field_shape_cfg.get("branching_root_y_frac", 0.0))
    electrode_points_all = field_shape_cfg.get("branching_seed_points_norm")
    electrode_points = field_shape_cfg.get("branching_seed_points_norm_right")
    if electrode_points_all is None:
        electrode_points_all = electrode_points

    return {
        "node_params": node_params,
        "node_count": node_count,
        "max_children": max_children,
        "depth_max": depth_max,
        "angle_frac": angle_frac,
        "child_angle_frac": child_angle_frac,
        "size_min_frac": min(size_min_frac, size_max_frac),
        "size_max_frac": max(size_min_frac, size_max_frac),
        "root_length_frac": max(1e-6, root_length_frac),
        "length_decay": float(np.clip(length_decay, 0.35, 0.98)),
        "width_frac": max(1e-6, width_frac),
        "width_decay": float(np.clip(width_decay, 0.35, 0.98)),
        "seed_all_electrodes": seed_all_electrodes,
        "center_x": center_x,
        "center_y": center_y,
        "root_x": root_x,
        "root_y": root_y,
        "electrode_points_all": _electrode_array(electrode_points_all),
        "electrode_points": _electrode_array(electrode_points),
    }


def _evaluate_branching_tree_field_from_normalized_coords(xn, yn, theta, field_shape_cfg, return_strength=False):
    params = _branching_tree_params_from_cfg(field_shape_cfg, theta)

    xn = np.asarray(xn, dtype=float)
    yn = np.asarray(yn, dtype=float)
    node_params = params["node_params"]
    node_count = params["node_count"]
    max_children = params["max_children"]
    depth_max = params["depth_max"]
    angle_frac = params["angle_frac"]
    child_angle_frac = params["child_angle_frac"]
    size_min_frac = params["size_min_frac"]
    size_max_frac = params["size_max_frac"]
    root_length_frac = params["root_length_frac"]
    length_decay = params["length_decay"]
    width_frac = params["width_frac"]
    width_decay = params["width_decay"]

    center_target = np.array([params["center_x"], params["center_y"]], dtype=float)
    if params["seed_all_electrodes"] and params["electrode_points_all"].size > 0:
        root_points = params["electrode_points_all"]
    elif params["electrode_points"].size > 0:
        root_points = params["electrode_points"]
    else:
        root_points = np.asarray([[params["root_x"], params["root_y"]]], dtype=float)

    segments = []

    def _grow(node_index, x0, y0, base_angle, base_length, depth):
        if node_index >= node_count or depth > depth_max:
            return

        angle_raw, size_raw, child_raw = node_params[node_index]
        node_angle = base_angle + (np.pi * angle_frac * float(angle_raw))
        size_scale = size_min_frac + 0.5 * (float(size_raw) + 1.0) * (size_max_frac - size_min_frac)
        segment_length = base_length * max(1e-6, size_scale)

        x1 = x0 + segment_length * np.cos(node_angle)
        y1 = y0 + segment_length * np.sin(node_angle)
        segments.append((x0, y0, x1, y1, depth))

        if depth >= depth_max or segment_length <= 1e-6:
            return

        child_count = int(np.clip(np.rint(0.5 * (float(child_raw) + 1.0) * max_children), 0, max_children))
        next_length = segment_length * length_decay
        child_angle_step = np.pi * child_angle_frac

        for child_slot in range(child_count):
            child_index = _branching_tree_child_index(node_index, child_slot, max_children)
            if child_index >= node_count:
                continue
            offset = child_slot - 0.5 * (child_count - 1)
            child_base_angle = node_angle + (offset * child_angle_step)
            _grow(child_index, x1, y1, child_base_angle, next_length, depth + 1)

    for root_idx, (rx, ry) in enumerate(np.asarray(root_points, dtype=float)):
        root_angle = float(np.arctan2(center_target[1] - ry, center_target[0] - rx))
        root_length = root_length_frac
        if root_idx == 0 and not params["seed_all_electrodes"]:
            root_length = float(np.hypot(rx - params["root_x"], ry - params["root_y"])) + 1e-6
        _grow(0, float(rx), float(ry), root_angle, root_length, 0)

    if not segments:
        field = -0.5 * np.ones_like(xn, dtype=float)
        if return_strength:
            return field, np.zeros_like(xn, dtype=float), 0.5
        return field

    strength = np.zeros_like(xn, dtype=float)
    for x0, y0, x1, y1, depth in segments:
        dist = _branching_segment_distance(xn, yn, x0, y0, x1, y1)
        width = max(0.015, width_frac * (width_decay ** depth))
        depth_weight = length_decay ** depth
        contribution = depth_weight * np.exp(-0.5 * (dist / width) ** 2)
        strength = np.maximum(strength, contribution)

    if np.max(strength) > 0.0:
        strength = strength / np.max(strength)

    field = strength - 0.5
    if return_strength:
        return field, strength, 0.5
    return field


def main():
    parser = argparse.ArgumentParser(description="Mesh generation and design smoke test.")
    parser.add_argument("--config", default=None, help="Optional INI config path.")
    parser.add_argument("--n-el", type=int, default=16, help="Number of electrodes (must be divisible by 4).")
    parser.add_argument("--h0", type=float, default=0.1, help="Target mesh size.")
    parser.add_argument(
        "--model",
        choices=["rbf", "fractal", "branching", "fungal"],
        default="rbf",
        help="Field parameterization model to prepare.",
    )
    parser.add_argument("--show-mesh", action="store_true", help="Force-enable mesh visualization.")
    parser.add_argument("--hide-mesh", action="store_true", help="Disable mesh visualization.")
    parser.add_argument(
        "--show-element-ids",
        action="store_true",
        help="Overlay triangle element indices on the mesh plot.",
    )
    args = parser.parse_args()

    cfg = {}
    if args.config:
        cfg = _read_mesh_generation_cfg(args.config)

    n_el = int(cfg.get("n_el", args.n_el)) if args.config else int(args.n_el)
    h0 = float(cfg.get("h0", args.h0)) if args.config else float(args.h0)
    model = str(cfg.get("model", args.model)) if args.config else str(args.model)

    show_mesh_cfg = bool(cfg.get("show_mesh", True)) if args.config else True
    show_ids_cfg = bool(cfg.get("show_element_ids", False)) if args.config else False
    show_mesh = (show_mesh_cfg and not args.hide_mesh) or args.show_mesh
    show_element_ids = show_ids_cfg or args.show_element_ids

    mesh_obj, protocol_obj, variable_mask, p1, p2 = build_model(n_el=n_el, h0=h0)

    field_shape_cfg = {
        "model": model,
        "rbf_rows": int(cfg.get("rbf_rows", 4)) if args.config else 4,
        "rbf_cols": int(cfg.get("rbf_cols", 5)) if args.config else 5,
        "rbf_sigma_frac": float(cfg.get("rbf_sigma_frac", 0.9)) if args.config else 0.9,
        "rbf_weight_scale": float(cfg.get("rbf_weight_scale", 0.35)) if args.config else 0.35,
        "fractal_iter_max": int(cfg.get("fractal_iter_max", 40)) if args.config else 40,
        "fractal_type": str(cfg.get("fractal_type", "branching")) if args.config else "branching",
        "fractal_power": int(cfg.get("fractal_power", 3)) if args.config else 3,
        "fractal_shift_frac": float(cfg.get("fractal_shift_frac", 0.55)) if args.config else 0.55,
        "fractal_center_origin": bool(cfg.get("fractal_center_origin", True)) if args.config else True,
        "fractal_threshold_init": float(cfg.get("fractal_threshold_init", 0.0)) if args.config else 0.0,
        "branching_depth_max": int(cfg.get("branching_depth_max", 5)) if args.config else 5,
        "branching_max_children": int(cfg.get("branching_max_children", 2)) if args.config else 2,
        "branching_child_angle_frac": float(cfg.get("branching_child_angle_frac", 0.34)) if args.config else 0.34,
        "branching_size_min_frac": float(cfg.get("branching_size_min_frac", 0.55)) if args.config else 0.55,
        "branching_size_max_frac": float(cfg.get("branching_size_max_frac", 1.0)) if args.config else 1.0,
    }
    geometry = build_field_geometry(mesh_obj, variable_mask, p1, p2, field_shape_cfg)

    print(f"Elements: {mesh_obj.element.shape[0]}")
    print(f"Electrodes: {len(mesh_obj.el_pos)}")
    print(f"Measurements: {protocol_obj.meas_mat.shape[0]}")
    print(f"Variable elements: {np.count_nonzero(variable_mask)}")
    print(f"Model: {model}")
    print(f"Geometry keys: {sorted(geometry.keys())}")

    if show_mesh:
        visualize_mesh(
            mesh_obj=mesh_obj,
            p1=p1,
            p2=p2,
            title=f"Generated mesh ({model})",
            show_element_ids=show_element_ids,
        )


if __name__ == "__main__":
    main()
