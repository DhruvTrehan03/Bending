from __future__ import absolute_import, division, print_function

import configparser
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, colormaps
import os
from scipy.optimize import differential_evolution

import pyeit.eit.protocol as protocol
import pyeit.mesh as mesh
from pyeit.mesh.shape import rectangle
from pyeit.eit.fem import EITForward


def _clean_cfg_string(value, default=""):
    """Return a normalized config string and strip inline comments."""
    if value is None:
        return str(default).strip().lower()
    text = str(value)
    for sep in ("#", ";"):
        if sep in text:
            text = text.split(sep, 1)[0]
    return text.strip().lower()


class LiveMeshViewer:
    """Real-time mesh viewer with VisPy backend and Matplotlib fallback."""

    def __init__(
        self,
        mesh_obj,
        p1,
        p2,
        low_cond,
        high_cond,
        backend="auto",
        field_shape_cfg=None,
    ):
        self.mesh_obj = mesh_obj
        self.p1 = p1
        self.p2 = p2
        self.low_cond = float(low_cond)
        self.high_cond = float(high_cond)
        self.backend = _clean_cfg_string(backend, default="auto")
        self.field_shape_cfg = dict(field_shape_cfg) if field_shape_cfg is not None else None

        self.mode = None
        self._matplotlib_handles = {}
        self._vispy_handles = {}
        self._init_backend()

    def _fractal_overlay_rgba(self, theta, grid_n=260):
        """Build a transparent RGBA overlay for the positive fractal region."""
        if self.field_shape_cfg is None:
            return None
        model = str(self.field_shape_cfg.get("model", "rbf")).strip().lower()
        if model != "fractal" or theta is None:
            return None

        _, _, field, strength, _ = evaluate_fractal_grid(
            theta=np.asarray(theta, dtype=float).ravel(),
            field_shape_cfg=self.field_shape_cfg,
            p1=self.p1,
            p2=self.p2,
            grid_n=grid_n,
            return_strength=True,
        )
        rgba = np.zeros((field.shape[0], field.shape[1], 4), dtype=np.float32)

        # Continuous fractal preview reveals boundary detail better than
        # a binary in/out fill while still highlighting the selected region.
        color_lut = colormaps.get_cmap("magma")
        rgb = color_lut(np.clip(strength, 0.0, 1.0))[..., :3]
        rgba[..., :3] = rgb.astype(np.float32)

        alpha = 0.04 + 0.20 * np.power(np.clip(strength, 0.0, 1.0), 1.25)
        alpha[field >= 0.0] = np.maximum(alpha[field >= 0.0], 0.26)
        rgba[..., 3] = alpha.astype(np.float32)

        return rgba

    def _perm_to_rgba(self, perm, cmap_name="RdBu_r"):
        perm = np.asarray(perm, dtype=float).ravel()
        if self.high_cond > self.low_cond > 0:
            norm = colors.LogNorm(vmin=self.low_cond, vmax=self.high_cond)
        else:
            norm = colors.Normalize(vmin=np.min(perm), vmax=np.max(perm))
        mapper = colormaps.get_cmap(cmap_name)
        return mapper(norm(perm))

    def _init_backend(self):
        try_vispy = self.backend in {"auto", "vispy"}
        if try_vispy:
            try:
                self._init_vispy()
                self.mode = "vispy"
                print("Live viewer backend: vispy")
                return
            except Exception as exc:
                if self.backend == "vispy":
                    print(f"VisPy unavailable ({exc}). Falling back to matplotlib.")
                else:
                    print(f"VisPy not used ({exc}); using matplotlib live viewer.")

        self._init_matplotlib()
        self.mode = "matplotlib"
        print("Live viewer backend: matplotlib")

    def _init_matplotlib(self):
        plt.ion()
        pts = self.mesh_obj.node
        tri = self.mesh_obj.element
        el_pos = self.mesh_obj.el_pos

        fig, ax = plt.subplots(figsize=(8, 4))
        im = ax.tripcolor(
            pts[:, 0],
            pts[:, 1],
            tri,
            np.full(tri.shape[0], self.low_cond, dtype=float),
            shading="flat",
            cmap="RdBu_r",
            norm=colors.LogNorm(vmin=self.low_cond, vmax=self.high_cond),
        )
        ax.plot(pts[el_pos, 0], pts[el_pos, 1], "ko", markersize=3)
        ax.set_aspect("equal")
        ax.set_xlim(self.p1[0] - 0.2, self.p2[0] + 0.2)
        ax.set_ylim(self.p1[1] - 0.2, self.p2[1] + 0.2)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Live Optimization Mesh")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Conductivity")
        _ = cbar
        fig.tight_layout()

        self._matplotlib_handles = {
            "fig": fig,
            "ax": ax,
            "im": im,
        }

    def _init_vispy(self):
        from vispy import app, scene
        from vispy.visuals.transforms import STTransform

        pts = self.mesh_obj.node
        tri = self.mesh_obj.element
        el_pos = self.mesh_obj.el_pos

        verts3 = np.column_stack([pts[:, 0], pts[:, 1], np.zeros(pts.shape[0], dtype=float)])
        init_colors = self._perm_to_rgba(np.full(tri.shape[0], self.low_cond, dtype=float))

        canvas = scene.SceneCanvas(
            keys="interactive",
            size=(1100, 650),
            show=True,
            title="Live Optimization Mesh",
            bgcolor="white",
        )
        view = canvas.central_widget.add_view()
        mesh_visual = scene.visuals.Mesh(
            vertices=verts3,
            faces=tri,
            face_colors=init_colors,
            shading=None,
            parent=view.scene,
        )

        fractal_image = None
        fractal_model_active = (
            self.field_shape_cfg is not None
            and str(self.field_shape_cfg.get("model", "rbf")).strip().lower() == "fractal"
        )
        if fractal_model_active:
            x1, y1 = self.p1
            x2, y2 = self.p2
            initial_rgba = np.zeros((260, 260, 4), dtype=np.float32)
            fractal_image = scene.visuals.Image(
                initial_rgba,
                interpolation="nearest",
                parent=view.scene,
            )
            fractal_image.transform = STTransform(
                scale=((x2 - x1) / 259.0, (y2 - y1) / 259.0, 1.0),
                translate=(x1, y1, 0.02),
            )

        scene.visuals.Markers(
            pos=verts3[el_pos],
            face_color="black",
            edge_color="black",
            size=8,
            parent=view.scene,
        )

        view.camera = scene.PanZoomCamera(aspect=1.0)
        view.camera.set_range(
            x=(self.p1[0] - 0.2, self.p2[0] + 0.2),
            y=(self.p1[1] - 0.2, self.p2[1] + 0.2),
            margin=0.0,
        )

        self._vispy_handles = {
            "app": app,
            "canvas": canvas,
            "view": view,
            "mesh": mesh_visual,
            "verts3": verts3,
            "tri": tri,
            "fractal_image": fractal_image,
            "fractal_model_active": fractal_model_active,
        }

    def _set_vispy_title(self, generation, best_score, current_score):
        title = (
            f"Live Optimization Mesh | Gen {generation} | "
            f"best={best_score:.6e} | current={current_score:.6e}"
        )
        canvas = self._vispy_handles["canvas"]
        try:
            canvas.title = title
        except Exception:
            try:
                canvas.native.setWindowTitle(title)
            except Exception:
                pass

    def update(self, perm, generation, best_score, current_score, theta=None):
        if self.mode == "vispy":
            colors_rgba = self._perm_to_rgba(perm)
            self._vispy_handles["mesh"].set_data(
                vertices=self._vispy_handles["verts3"],
                faces=self._vispy_handles["tri"],
                face_colors=colors_rgba,
            )
            if self._vispy_handles.get("fractal_model_active", False):
                rgba = self._fractal_overlay_rgba(theta=theta, grid_n=260)
                if rgba is not None:
                    self._vispy_handles["fractal_image"].set_data(rgba)
            self._set_vispy_title(generation, best_score, current_score)
            self._vispy_handles["canvas"].update()
            self._vispy_handles["app"].process_events()
            return

        if self.mode == "matplotlib":
            im = self._matplotlib_handles["im"]
            ax = self._matplotlib_handles["ax"]
            fig = self._matplotlib_handles["fig"]
            im.set_array(np.asarray(perm, dtype=float).ravel())
            ax.set_title(
                f"Live Optimization Mesh | Gen {generation} | "
                f"best={best_score:.6e} | current={current_score:.6e}"
            )
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.001)

    def close(self):
        if self.mode == "vispy" and self._vispy_handles:
            try:
                self._vispy_handles["canvas"].close()
            except Exception:
                pass
        if self.mode == "matplotlib" and self._matplotlib_handles:
            try:
                plt.close(self._matplotlib_handles["fig"])
            except Exception:
                pass


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


def build_model(n_el=16, h0=0.1):
    """Build a plain 5x2 rectangle model with all elements optimizable."""
    p1, p2 = [0.0, 0.0], [5.0, 2.0]
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


def build_variable_element_geometry(mesh_obj, variable_mask, p1, p2):
    """Precompute geometry used by the parameterized field-shape model."""
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

    # Distances to rectangle sides: bottom, right, top, left.
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
    """Map each electrode to nearby variable elements for coverage constraints."""
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
    """Return number of optimized parameters for the RBF level-set model."""
    return 1 + int(rbf_rows) * int(rbf_cols)


def parameter_vector_size_from_cfg(field_shape_cfg, field_geometry=None):
    """Return parameter count for the selected field-shape model."""
    model = str(field_shape_cfg.get("model", "rbf")).strip().lower()
    if model == "rbf":
        return parameter_vector_size(
            int(field_shape_cfg["rbf_rows"]),
            int(field_shape_cfg["rbf_cols"]),
        )
    if model == "fractal":
        # Center-locked: threshold, c_real, c_imag, zoom_log2, angle
        # Free-shift: threshold, c_real, c_imag, zoom_log2, shift_x, shift_y, angle
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
    """Return validated fractal type from config."""
    fractal_type = _clean_cfg_string(field_shape_cfg.get("fractal_type", "julia"), default="julia")
    allowed = {"julia", "mandelbrot", "burning_ship", "multibrot", "branching"}
    if fractal_type not in allowed:
        raise ValueError(
            f"Unsupported field_shape.fractal_type '{fractal_type}'. "
            "Use 'julia', 'mandelbrot', 'burning_ship', 'multibrot', or 'branching'."
        )
    return fractal_type


def _fractal_power_from_cfg(field_shape_cfg):
    """Return exponent for multibrot family (minimum 2)."""
    return int(max(2, int(field_shape_cfg.get("fractal_power", 3))))


def _wrap_angle(angle):
    """Wrap angle to the range [-pi, pi]."""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def _blend_angles(angle_a, angle_b, blend):
    """Blend two angles using the shortest angular direction."""
    blend = float(np.clip(blend, 0.0, 1.0))
    delta = _wrap_angle(angle_b - angle_a)
    return _wrap_angle(angle_a + blend * delta)


def _nearest_electrode_angle(x, y, electrode_points):
    """Return angle from (x, y) to the nearest electrode point."""
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
    """Normalize optional electrode-point input to a Nx2 array."""
    if electrode_points is None:
        return np.empty((0, 2), dtype=float)
    arr = np.asarray(electrode_points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        return np.empty((0, 2), dtype=float)
    return arr


def _branching_segment_distance(xn, yn, x0, y0, x1, y1):
    """Return Euclidean distance from points to a line segment."""
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
    """Return deterministic pseudo-random value in [-1, 1] from numeric inputs."""
    seed = 0.0
    coeff = 0.7548776662466927
    for val in values:
        seed = seed * 1.618033988749895 + coeff * float(val)
        coeff = coeff * 1.4142135623730951 + 0.2718281828459045
    raw = np.sin(seed * 12.9898 + 78.233) * 43758.5453123
    frac = raw - np.floor(raw)
    return float(2.0 * frac - 1.0)


def _branching_fractal_params_from_cfg(field_shape_cfg, theta):
    """Extract branching-fractal parameters from config and theta."""
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

    # The theta values still influence the exact growth geometry so the
    # optimizer can steer the branch structure rather than only the threshold.
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


def _evaluate_branching_fractal_field_from_normalized_coords(
    xn,
    yn,
    theta,
    field_shape_cfg,
    return_strength=False,
):
    """Evaluate a recursive branching fractal field on normalized coordinates."""
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
        """Deterministic angular jitter that grows near the center target."""
        if params["random_angle"] <= 0.0:
            return 0.0

        dist_to_center = np.hypot(x_pos - center_target[0], y_pos - center_target[1])
        center_proximity = 1.0 - np.clip(dist_to_center / root_radius, 0.0, 1.0)
        center_gain = 1.0 + params["random_center_boost"] * (
            center_proximity ** params["random_center_power"]
        )
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
        """Blend the current branch heading toward the nearest electrode."""
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
        # Force geometric contact by attaching each target electrode to the
        # nearest existing branch point on the generated half-tree.
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


def _evaluate_fractal_field_from_normalized_coords(
    xn,
    yn,
    theta,
    field_shape_cfg,
    return_strength=False,
):
    """Evaluate configured fractal field on normalized coordinates."""
    fractal_type = _fractal_type_from_cfg(field_shape_cfg)
    if fractal_type == "branching":
        return _evaluate_branching_fractal_field_from_normalized_coords(
            xn=xn,
            yn=yn,
            theta=theta,
            field_shape_cfg=field_shape_cfg,
            return_strength=return_strength,
        )

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
    max_iter = int(max(1, field_shape_cfg["fractal_iter_max"]))

    multibrot_power = _fractal_power_from_cfg(field_shape_cfg)

    x0 = xn + shift_x
    y0 = yn + shift_y
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    xr = zoom * (x0 * cos_a - y0 * sin_a)
    yi = zoom * (x0 * sin_a + y0 * cos_a)

    c_offset_scale = 0.35
    if fractal_type == "julia":
        zr = np.asarray(xr, dtype=float).copy()
        zi = np.asarray(yi, dtype=float).copy()
        cr = np.full_like(zr, c_real, dtype=float)
        ci = np.full_like(zi, c_imag, dtype=float)
        power = 2
    elif fractal_type in {"mandelbrot", "burning_ship"}:
        zr = np.zeros_like(xr, dtype=float)
        zi = np.zeros_like(yi, dtype=float)
        cr = np.asarray(xr, dtype=float) + c_offset_scale * c_real
        ci = np.asarray(yi, dtype=float) + c_offset_scale * c_imag
        power = 2
    else:  # multibrot
        zr = np.zeros_like(xr, dtype=float)
        zi = np.zeros_like(yi, dtype=float)
        cr = np.asarray(xr, dtype=float) + c_offset_scale * c_real
        ci = np.asarray(yi, dtype=float) + c_offset_scale * c_imag
        power = multibrot_power

    escaped_at = np.full(zr.shape, max_iter, dtype=int)
    alive = np.ones(zr.shape, dtype=bool)

    for it in range(max_iter):
        if not np.any(alive):
            break

        zr_a = zr[alive]
        zi_a = zi[alive]
        cr_a = cr[alive]
        ci_a = ci[alive]

        if fractal_type == "burning_ship":
            ar = np.abs(zr_a)
            ai = np.abs(zi_a)
            zr_next = ar * ar - ai * ai + cr_a
            zi_next = 2.0 * ar * ai + ci_a
        elif power == 2:
            zr_next = zr_a * zr_a - zi_a * zi_a + cr_a
            zi_next = 2.0 * zr_a * zi_a + ci_a
        else:
            z_pow = (zr_a + 1j * zi_a) ** power
            zr_next = np.real(z_pow) + cr_a
            zi_next = np.imag(z_pow) + ci_a

        zr[alive] = zr_next
        zi[alive] = zi_next
        escaped = (zr_next * zr_next + zi_next * zi_next) > 4.0

        if np.any(escaped):
            alive_flat = np.flatnonzero(alive)
            escaped_flat = alive_flat[escaped]
            escaped_at.flat[escaped_flat] = it
            alive.flat[escaped_flat] = False

    fractal_strength = 1.0 - (escaped_at.astype(float) / float(max_iter))
    field = fractal_strength - threshold
    if return_strength:
        return field, fractal_strength, threshold
    return field


def build_rbf_field_geometry(mesh_obj, variable_mask, p1, p2, rbf_rows, rbf_cols, sigma_frac):
    """Precompute RBF centers and basis values for the implicit shape model."""
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
    """Precompute normalized coordinates for fractal field parameterization."""
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
    # Use a single spatial scale so fractal geometry is not stretched by
    # rectangular aspect ratio in physical coordinates.
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


def build_field_geometry(mesh_obj, variable_mask, p1, p2, field_shape_cfg):
    """Build geometry cache for the active field-shape model."""
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
    if model == "element":
        # Element model: one parameter per variable element
        n_variable_elements = int(np.sum(variable_mask))
        return {
            "n_variable_elements": n_variable_elements,
        }
    raise ValueError(f"Unsupported field-shape model: {model}")


def parameterized_state_from_theta(theta, field_geometry, field_shape_cfg):
    """Convert RBF parameters into a boolean high-conductivity state."""
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

    if model == "element":
        # Element model: each parameter directly determines if element is high (param > 0) or low (param <= 0)
        params_clipped = np.clip(theta, -1.0, 1.0)
        state_high = params_clipped >= 0.0
        return np.asarray(state_high, dtype=bool), params_clipped

    raise ValueError(f"Unsupported field-shape model: {model}")


def evaluate_fractal_grid(theta, field_shape_cfg, p1, p2, grid_n=360, return_strength=False):
    """Evaluate fractal level-set field on a dense plotting grid."""
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


def make_permittivity(state_high, variable_mask, low_cond, high_cond):
    """Construct element conductivity from a boolean high/low element state."""
    n_elem = variable_mask.size
    perm = np.full(n_elem, low_cond, dtype=float)

    variable_indices = np.where(variable_mask)[0]
    perm[variable_indices[state_high]] = high_cond
    return perm


def element_sensitivity_from_jacobian(jacobian):
    """Calculate element sensitivity as the L1 norm of jacobian columns."""
    return np.log10(np.sum(np.abs(jacobian), axis=0) + np.finfo(float).tiny)


def entropy_score(sensitivity):
    """Compute Shannon entropy of element sensitivity distribution.
    
    Lower entropy indicates more uniform/concentrated sensitivity across elements.
    Returns the Shannon entropy: -sum(p * log(p)) where p is a normalized
    probability distribution of sensitivity values.
    """
    vals = np.abs(np.asarray(sensitivity).ravel())
    eps = np.finfo(float).eps
    
    total = np.sum(vals)
    if total < eps:
        return 0.0
    
    probabilities = vals / total
    # Only include non-zero probabilities to avoid log(0)
    entropy = -np.sum(probabilities[probabilities > eps] * np.log(probabilities[probabilities > eps]))
    mean_sensitivity = np.mean(vals)
    return float(entropy+mean_sensitivity)


def build_element_adjacency(mesh_obj):
    """Build adjacency matrix for elements based on shared vertices.
    
    Two elements are considered adjacent if they share at least 2 vertices (an edge).
    Returns adjacency as a list of lists where adjacency[i] contains indices of
    elements adjacent to element i.
    """
    tri = mesh_obj.element
    n_elem = tri.shape[0]
    adjacency = [[] for _ in range(n_elem)]
    
    for i in range(n_elem):
        vertices_i = set(tri[i])
        for j in range(i + 1, n_elem):
            vertices_j = set(tri[j])
            # Elements are adjacent if they share at least 2 vertices (an edge)
            if len(vertices_i & vertices_j) >= 2:
                adjacency[i].append(j)
                adjacency[j].append(i)
    
    return adjacency


def get_connected_components(state_high, adjacency):
    """Identify connected components of high-conductivity elements.
    
    Args:
        state_high: Boolean array where True indicates high-conductivity element
        adjacency: Adjacency list where adjacency[i] is list of adjacent elements
    
    Returns:
        List of lists, where each sublist contains indices of elements in a
        connected component of high-conductivity elements.
    """
    n_elem = len(state_high)
    visited = np.zeros(n_elem, dtype=bool)
    components = []
    
    for start_elem in range(n_elem):
        if not state_high[start_elem] or visited[start_elem]:
            continue
        
        # BFS to find all connected high-conductivity elements
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
    """Count number of isolated high-conductivity elements.
    
    An element is isolated if it has no high-conductivity neighbors.
    """
    isolated_count = 0
    for i in range(len(state_high)):
        if state_high[i]:
            has_high_neighbor = any(state_high[j] for j in adjacency[i])
            if not has_high_neighbor:
                isolated_count += 1
    
    return isolated_count


def has_connected_high_elements(state_high, adjacency):
    """Check if all high-conductivity elements form a single connected component.
    
    Returns True if all high-conductivity elements are connected, False otherwise.
    If there are no high-conductivity elements or only one, returns True.
    """
    n_high = np.sum(state_high)
    if n_high <= 1:
        return True
    
    components = get_connected_components(state_high, adjacency)
    return len(components) == 1


def electrodes_connected_by_high_region(state_high, adjacency, electrode_supports):
    """Check that one connected high-conductivity component touches every electrode."""
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


def enforce_connected_high_elements(state_high, adjacency):
    """Repair state so high-conductivity elements form at most one component.

    If multiple high components exist, only the largest component is kept and
    all other high elements are set to low.
    """
    state = np.asarray(state_high, dtype=bool).copy()
    components = get_connected_components(state, adjacency)
    if len(components) <= 1:
        return state

    keep = max(components, key=len)
    repaired = np.zeros_like(state, dtype=bool)
    repaired[np.asarray(keep, dtype=int)] = True
    return repaired


def evaluate_state(
    fwd,
    state,
    variable_mask,
    low_cond,
    high_cond,
    adjacency=None,
    electrode_supports=None,
):
    """Evaluate element state by computing sensitivity entropy with connectivity constraint.
    
    Args:
        fwd: Forward EIT solver
        state: Boolean array of high/low conductivity state
        variable_mask: Mask of variable elements
        low_cond: Low conductivity value
        high_cond: High conductivity value
        adjacency: Element adjacency list (optional). If provided, penalty applied
            when high-conductivity elements split into multiple components.
    """
    perm = make_permittivity(
        state_high=state,
        variable_mask=variable_mask,
        low_cond=low_cond,
        high_cond=high_cond,
    )

    # Apply strict connectivity constraint.
    # High elements must form one connected component.
    if adjacency is not None and not electrodes_connected_by_high_region(
        state,
        adjacency,
        electrode_supports,
    ):
            sensitivity = np.zeros_like(perm, dtype=float)
            return perm, sensitivity, np.inf

    jacobian, _ = fwd.compute_jac(perm=perm)
    sensitivity = element_sensitivity_from_jacobian(jacobian)
    score = entropy_score(sensitivity)

    return perm, sensitivity, score


def optimize_parameterized_field_ga(
    fwd,
    variable_mask,
    field_geometry,
    field_shape_cfg,
    low_cond,
    high_cond,
    generations,
    pop_size,
    elite_count,
    crossover_rate,
    mutation_rate,
    init_high_fraction,
    tournament_size,
    seed,
    adjacency=None,
    electrode_supports=None,
    generation_callback=None,
    live_update_every=1,
):
    """Optimize parameterized conductivity-field shape with a genetic algorithm."""
    rng = np.random.default_rng(seed)

    model = str(field_shape_cfg.get("model", "rbf")).strip().lower()
    n_params = parameter_vector_size_from_cfg(field_shape_cfg)
    generations = int(max(1, generations))
    pop_size = int(max(4, pop_size))
    elite_count = int(max(1, min(elite_count, pop_size - 1)))
    crossover_rate = float(min(max(crossover_rate, 0.0), 1.0))
    mutation_rate = float(min(max(mutation_rate, 0.0), 1.0))
    init_high_fraction = float(min(max(init_high_fraction, 0.0), 1.0))
    tournament_size = int(max(2, min(tournament_size, pop_size)))
    live_update_every = int(max(1, live_update_every))

    population = rng.normal(loc=0.0, scale=0.25, size=(pop_size, n_params))
    if model == "rbf":
        population[:, 0] = np.clip(
            rng.normal(loc=float(field_shape_cfg["rbf_bias_init"]), scale=0.20, size=pop_size),
            -1.0,
            1.0,
        )
        population[:, 1:] = np.clip(
            rng.normal(
                loc=0.0,
                scale=float(field_shape_cfg["rbf_weight_scale"]),
                size=(pop_size, n_params - 1),
            ),
            -1.0,
            1.0,
        )
    elif model == "fractal":
        population = np.clip(rng.normal(loc=0.0, scale=0.40, size=(pop_size, n_params)), -1.0, 1.0)
        population[:, 0] = np.clip(
            rng.normal(loc=float(field_shape_cfg["fractal_threshold_init"]), scale=0.25, size=pop_size),
            -1.0,
            1.0,
        )
    else:
        raise ValueError(f"Unsupported field-shape model: {model}")

    best_state = None
    best_theta = None
    best_thickness = None
    best_perm = None
    best_sensitivity = None
    best_score = np.inf

    best_history = []
    current_history = []

    for gen in range(1, generations + 1):
        evaluated = []
        for i in range(pop_size):
            theta = population[i]
            state, thickness = parameterized_state_from_theta(
                theta=theta,
                field_geometry=field_geometry,
                field_shape_cfg=field_shape_cfg,
            )
            perm, sensitivity, score = evaluate_state(
                fwd=fwd,
                state=state,
                variable_mask=variable_mask,
                low_cond=low_cond,
                high_cond=high_cond,
                adjacency=adjacency,
                electrode_supports=electrode_supports,
            )
            evaluated.append({
                "theta": theta.copy(),
                "state": state.copy(),
                "field": np.asarray(thickness).copy(),
                "perm": perm,
                "sensitivity": np.asarray(sensitivity).copy(),
                "score": float(score),
            })

        evaluated.sort(key=lambda d: d["score"])
        gen_best = evaluated[0]
        gen_best_score = gen_best["score"]
        gen_mean_score = float(np.mean([e["score"] for e in evaluated]))

        if best_state is None or gen_best_score <= best_score:
            best_score = gen_best_score
            best_theta = gen_best["theta"].copy()
            best_state = gen_best["state"].copy()
            best_thickness = gen_best["field"].copy()
            best_perm = gen_best["perm"].copy()
            best_sensitivity = gen_best["sensitivity"].copy()

        best_history.append(float(best_score))
        current_history.append(float(gen_best_score))

        should_update = (
            generation_callback is not None
            and (gen == 1 or gen == generations or (gen % live_update_every) == 0)
        )
        if should_update:
            generation_callback(
                perm=gen_best["perm"],
                generation=gen,
                best_score=float(best_score),
                current_score=float(gen_best_score),
                theta=gen_best["theta"],
            )

        if gen % 10 == 0 or gen == generations:
            print(
                f"Gen {gen:4d}/{generations} | best_gen={gen_best_score:.6e} | "
                f"best_global={best_score:.6e} | mean={gen_mean_score:.6e}"
            )

        next_population = []

        # Elitism: carry top individuals unchanged
        for i in range(elite_count):
            next_population.append(evaluated[i]["theta"].copy())

        # Tournament selection and reproduction
        def tournament_select():
            idxs = rng.choice(pop_size, size=tournament_size, replace=False)
            winner = min(idxs, key=lambda idx: evaluated[idx]["score"])
            return evaluated[winner]["theta"].copy()

        while len(next_population) < pop_size:
            parent_a = tournament_select()
            parent_b = tournament_select()

            # Crossover
            if rng.random() < crossover_rate and n_params > 1:
                blend = rng.random(n_params)
                child = blend * parent_a + (1.0 - blend) * parent_b
            else:
                child = parent_a.copy()

            # Mutation
            if mutation_rate > 0.0:
                mutation_mask = rng.random(n_params) < mutation_rate
                if np.any(mutation_mask):
                    child[mutation_mask] += rng.normal(loc=0.0, scale=0.25, size=np.count_nonzero(mutation_mask))
            child = np.clip(child, -1.0, 1.0)

            next_population.append(child)

        population = np.asarray(next_population, dtype=float)

    return {
        "best_theta": best_theta,
        "best_state": best_state,
        "best_field": best_thickness,
        "best_perm": best_perm,
        "best_sensitivity": best_sensitivity,
        "best_score": float(best_score),
        "best_history": np.asarray(best_history, dtype=float),
        "current_history": np.asarray(current_history, dtype=float),
        "n_variable": int(np.count_nonzero(variable_mask)),
        "n_params": int(n_params),
    }


def optimize_parameterized_field_de(
    fwd,
    variable_mask,
    field_geometry,
    field_shape_cfg,
    low_cond,
    high_cond,
    maxiter,
    popsize,
    mutation,
    recombination,
    seed,
    adjacency=None,
    electrode_supports=None,
    generation_callback=None,
    live_update_every=1,
):
    """Optimize parameterized conductivity-field shape with Differential Evolution."""
    n_params = parameter_vector_size_from_cfg(field_shape_cfg)
    maxiter = int(max(1, maxiter))
    popsize = int(max(2, popsize))
    recombination = float(np.clip(recombination, 0.0, 1.0))
    live_update_every = int(max(1, live_update_every))

    if np.isscalar(mutation):
        mutation = float(np.clip(mutation, 0.0, 2.0))
    else:
        mutation = tuple(float(np.clip(v, 0.0, 2.0)) for v in mutation)
        if len(mutation) != 2:
            raise ValueError("de_mutation tuple must contain exactly 2 values")

    bounds = [(-1.0, 1.0)] * n_params

    best_state = None
    best_theta = None
    best_field = None
    best_perm = None
    best_sensitivity = None
    best_score = np.inf

    best_history = []
    current_history = []
    eval_count = 0

    def evaluate_theta(theta):
        nonlocal best_state, best_theta, best_field, best_perm, best_sensitivity, best_score, eval_count

        state, field = parameterized_state_from_theta(
            theta=theta,
            field_geometry=field_geometry,
            field_shape_cfg=field_shape_cfg,
        )
        perm, sensitivity, score = evaluate_state(
            fwd=fwd,
            state=state,
            variable_mask=variable_mask,
            low_cond=low_cond,
            high_cond=high_cond,
            adjacency=adjacency,
            electrode_supports=electrode_supports,
        )
        eval_count += 1

        if best_state is None or score <= best_score:
            best_score = float(score)
            best_theta = np.asarray(theta, dtype=float).copy()
            best_state = np.asarray(state, dtype=bool).copy()
            best_field = np.asarray(field, dtype=float).copy()
            best_perm = np.asarray(perm, dtype=float).copy()
            best_sensitivity = np.asarray(sensitivity, dtype=float).copy()

        return float(score)

    generation_counter = {"value": 0}

    def de_callback(xk, convergence):
        _ = convergence
        generation_counter["value"] += 1
        gen = generation_counter["value"]

        current_score = evaluate_theta(xk)
        current_history.append(float(current_score))
        best_history.append(float(best_score))

        should_update = (
            generation_callback is not None
            and (gen == 1 or gen == maxiter or (gen % live_update_every) == 0)
            and best_perm is not None
        )
        if should_update:
            generation_callback(
                perm=best_perm,
                generation=gen,
                best_score=float(best_score),
                current_score=float(current_score),
                theta=best_theta,
            )

        if gen % 10 == 0 or gen == maxiter:
            print(
                f"DE gen {gen:4d}/{maxiter} | best_global={best_score:.6e} | "
                f"current={current_score:.6e} | evals={eval_count}"
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
        # Safety fallback for edge cases where callback didn't run.
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
    """Show best conductivity map, sensitivity distribution, and optimization progress."""
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

    fig.suptitle("Global Binary Optimization (Entropy Objective)", fontsize=13)

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
            and str(field_shape_cfg.get("model", "rbf")).strip().lower() == "fractal"
            and best_theta is not None
        ):
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
        if r == 0 and field_shape_cfg is not None and str(field_shape_cfg.get("model", "rbf")).strip().lower() == "fractal":
            ax0.set_title(f"{row['name']} Conductivity + Fractal Overlay")
        else:
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
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label="||J[:, elem]||2")

        if r == 0:
            ax2.plot(best_hist, label="Best-so-far score", linewidth=1.8)
            ax2.plot(curr_hist, label="Current score", linewidth=1.0, alpha=0.7)
            ax2.set_title("Optimization Progress")
            ax2.set_xlabel("Iteration")
            ax2.set_ylabel("Entropy Score")
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc="best")
        else:
            ax2.axis("off")
            ax2.text(
                0.03,
                0.85,
                f"{row['name']}\nEntropy: {row['score']:.6e}\n"
                f"Sensitivity min: {np.min(sens_row):.6e}\n"
                f"Sensitivity max: {np.max(sens_row):.6e}",
                va="top",
                ha="left",
                fontsize=10,
            )

    plt.tight_layout()
    plt.show()




def load_config(config_file="config.ini"):
    """Load configuration from INI file."""
    config = configparser.ConfigParser()
    
    if not os.path.exists(config_file):
        print(f"Config file {config_file} not found. Creating with defaults...")
        create_default_config(config_file)
    
    config.read(config_file)
    
    cfg = type('Config', (), {})()
    
    # Optimization settings
    cfg.generations = config.getint('optimization', 'generations', fallback=100)
    cfg.pop_size = config.getint('optimization', 'pop_size', fallback=24)
    cfg.elite_count = config.getint('optimization', 'elite_count', fallback=2)
    cfg.crossover_rate = config.getfloat('optimization', 'crossover_rate', fallback=0.8)
    cfg.mutation_rate = config.getfloat('optimization', 'mutation_rate', fallback=0.01)
    cfg.tournament_size = config.getint('optimization', 'tournament_size', fallback=3)
    cfg.optimizer = _clean_cfg_string(config.get('optimization', 'optimizer', fallback='ga'), default='ga')
    cfg.de_maxiter = config.getint('optimization', 'de_maxiter', fallback=25)
    cfg.de_popsize = config.getint('optimization', 'de_popsize', fallback=6)
    cfg.de_mutation_min = config.getfloat('optimization', 'de_mutation_min', fallback=0.5)
    cfg.de_mutation_max = config.getfloat('optimization', 'de_mutation_max', fallback=1.0)
    cfg.de_recombination = config.getfloat('optimization', 'de_recombination', fallback=0.7)
    
    # Initial conditions
    cfg.init_high_fraction = config.getfloat('initial_conditions', 'init_high_fraction', fallback=0.5)
    cfg.seed = config.getint('initial_conditions', 'seed', fallback=7)
    
    # Conductivity
    cfg.low_cond = config.getfloat('conductivity', 'low_cond', fallback=1e-6)
    cfg.high_cond = config.getfloat('conductivity', 'high_cond', fallback=1e6)
    
    # Mesh
    cfg.h0 = config.getfloat('mesh', 'h0', fallback=0.1)

    # Field-shape parameterization
    cfg.field_model = _clean_cfg_string(config.get('field_shape', 'model', fallback='rbf'), default='rbf')
    cfg.rbf_rows = config.getint('field_shape', 'rbf_rows', fallback=4)
    cfg.rbf_cols = config.getint('field_shape', 'rbf_cols', fallback=5)
    cfg.rbf_sigma_frac = config.getfloat('field_shape', 'rbf_sigma_frac', fallback=0.90)
    cfg.rbf_bias_init = config.getfloat('field_shape', 'rbf_bias_init', fallback=0.25)
    cfg.rbf_weight_scale = config.getfloat('field_shape', 'rbf_weight_scale', fallback=0.35)
    cfg.fractal_iter_max = config.getint('field_shape', 'fractal_iter_max', fallback=40)
    cfg.fractal_type = _clean_cfg_string(
        config.get('field_shape', 'fractal_type', fallback='julia'),
        default='julia',
    )
    fractal_power_str = _clean_cfg_string(
        config.get('field_shape', 'fractal_power', fallback='3'),
        default='3',
    )
    cfg.fractal_power = int(float(fractal_power_str))
    cfg.fractal_shift_frac = config.getfloat('field_shape', 'fractal_shift_frac', fallback=0.55)
    cfg.fractal_threshold_init = config.getfloat('field_shape', 'fractal_threshold_init', fallback=0.0)
    cfg.fractal_center_origin = config.getboolean('field_shape', 'fractal_center_origin', fallback=True)
    cfg.branching_depth_max = config.getint('field_shape', 'branching_depth_max', fallback=6)
    cfg.branching_angle_frac = config.getfloat('field_shape', 'branching_angle_frac', fallback=0.34)
    cfg.branching_length_decay = config.getfloat('field_shape', 'branching_length_decay', fallback=0.68)
    cfg.branching_width_frac = config.getfloat('field_shape', 'branching_width_frac', fallback=0.12)
    cfg.branching_width_decay = config.getfloat('field_shape', 'branching_width_decay', fallback=0.80)
    cfg.branching_mirror_vertical = config.getboolean('field_shape', 'branching_mirror_vertical', fallback=False)
    cfg.branching_aim_electrodes = config.getboolean('field_shape', 'branching_aim_electrodes', fallback=False)
    cfg.branching_target_blend = config.getfloat('field_shape', 'branching_target_blend', fallback=0.0)
    cfg.branching_root_x_frac = config.getfloat('field_shape', 'branching_root_x_frac', fallback=0.18)
    cfg.branching_root_y_frac = config.getfloat('field_shape', 'branching_root_y_frac', fallback=0.0)
    cfg.branching_seed_all_electrodes = config.getboolean(
        'field_shape',
        'branching_seed_all_electrodes',
        fallback=True,
    )
    cfg.branching_force_meet_center = config.getboolean(
        'field_shape',
        'branching_force_meet_center',
        fallback=True,
    )
    cfg.branching_meet_x_frac = config.getfloat('field_shape', 'branching_meet_x_frac', fallback=0.0)
    cfg.branching_meet_y_frac = config.getfloat('field_shape', 'branching_meet_y_frac', fallback=0.0)
    cfg.branching_meet_blend = config.getfloat('field_shape', 'branching_meet_blend', fallback=1.0)
    cfg.branching_random_angle_frac = config.getfloat('field_shape', 'branching_random_angle_frac', fallback=0.00)
    cfg.branching_random_center_boost = config.getfloat('field_shape', 'branching_random_center_boost', fallback=0.00)
    cfg.branching_random_center_power = config.getfloat('field_shape', 'branching_random_center_power', fallback=2.0)
    cfg.branching_force_touch_all_electrodes = config.getboolean(
        'field_shape',
        'branching_force_touch_all_electrodes',
        fallback=True,
    )
    cfg.electrode_support_k = config.getint('field_shape', 'electrode_support_k', fallback=4)

    # Live visualization settings
    cfg.live_enabled = config.getboolean('visualization', 'live_enabled', fallback=True)
    cfg.live_backend = _clean_cfg_string(config.get('visualization', 'live_backend', fallback='auto'), default='auto')
    cfg.live_update_every = config.getint('visualization', 'live_update_every', fallback=2)

    if cfg.live_backend not in {"auto", "vispy", "matplotlib"}:
        raise ValueError(
            f"Unsupported visualization.live_backend '{cfg.live_backend}'. "
            "Use 'auto', 'vispy', or 'matplotlib'."
        )

    if cfg.field_model not in {"rbf", "fractal"}:
        raise ValueError(
            f"Unsupported field_shape.model '{cfg.field_model}'. Use 'rbf' or 'fractal'."
        )

    if cfg.fractal_type not in {"julia", "mandelbrot", "burning_ship", "multibrot", "branching"}:
        raise ValueError(
            f"Unsupported field_shape.fractal_type '{cfg.fractal_type}'. "
            "Use 'julia', 'mandelbrot', 'burning_ship', 'multibrot', or 'branching'."
        )

    if cfg.fractal_power < 2:
        raise ValueError("field_shape.fractal_power must be >= 2.")

    if cfg.optimizer not in {"ga", "de"}:
        raise ValueError(
            f"Unsupported optimization.optimizer '{cfg.optimizer}'. Use 'ga' or 'de'."
        )
    
    return cfg

def create_default_config(config_file="config.ini"):
    """Create a default configuration file."""
    config = configparser.ConfigParser()
    
    config['optimization'] = {
        'generations': '100',
        'pop_size': '24',
        'elite_count': '2',
        'crossover_rate': '0.8',
        'mutation_rate': '0.01',
        'tournament_size': '3',
        'optimizer': 'de',
        'de_maxiter': '25',
        'de_popsize': '6',
        'de_mutation_min': '0.5',
        'de_mutation_max': '1.0',
        'de_recombination': '0.7',
    }
    
    config['initial_conditions'] = {
        'init_high_fraction': '0.5',
        'seed': '7',
    }
    
    config['conductivity'] = {
        'low_cond': '100',
        'high_cond': '10000',
    }
    
    config['mesh'] = {
        'h0': '0.3',
    }

    config['field_shape'] = {
        'model': 'rbf',
        'rbf_rows': '4',
        'rbf_cols': '5',
        'rbf_sigma_frac': '0.90',
        'rbf_bias_init': '0.25',
        'rbf_weight_scale': '0.35',
        'fractal_iter_max': '40',
        'fractal_type': 'julia',
        'fractal_power': '3',
        'fractal_shift_frac': '0.55',
        'fractal_threshold_init': '0.00',
        'fractal_center_origin': 'true',
        'branching_depth_max': '6',
        'branching_angle_frac': '0.34',
        'branching_length_decay': '0.68',
        'branching_width_frac': '0.12',
        'branching_width_decay': '0.80',
        'branching_mirror_vertical': 'false',
        'branching_aim_electrodes': 'false',
        'branching_target_blend': '0.00',
        'branching_root_x_frac': '0.18',
        'branching_root_y_frac': '0.0',
        'branching_seed_all_electrodes': 'true',
        'branching_force_meet_center': 'true',
        'branching_meet_x_frac': '0.0',
        'branching_meet_y_frac': '0.0',
        'branching_meet_blend': '1.0',
        'branching_random_angle_frac': '0.35',
        'branching_random_center_boost': '0.50',
        'branching_random_center_power': '2.0',
        'branching_force_touch_all_electrodes': 'true',
        'electrode_support_k': '4',
    }

    config['visualization'] = {
        'live_enabled': 'true',
        'live_backend': 'auto',
        'live_update_every': '2',
    }
    
    with open(config_file, 'w') as f:
        config.write(f)

def run():
    # Load configuration from file
    config_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(config_dir, "config.ini")
    args = load_config(config_file)

    mesh_obj, protocol_obj, variable_mask, p1, p2 = build_model(h0=args.h0)
    fwd = EITForward(mesh_obj, protocol_obj)

    print(f"Total elements: {mesh_obj.element.shape[0]}")
    print(f"Optimized variable elements: {np.count_nonzero(variable_mask)}")

    # Build topology used for strict conductivity-path constraints.
    adjacency = build_element_adjacency(mesh_obj)
    electrode_supports = build_electrode_support_elements(
        mesh_obj,
        variable_mask,
        k_nearest=args.electrode_support_k,
    )
    print(f"Element adjacency built (mesh has {len(adjacency)} elements)")

    field_shape_cfg = {
        "model": str(args.field_model).strip().lower(),
        "rbf_rows": int(max(1, args.rbf_rows)),
        "rbf_cols": int(max(1, args.rbf_cols)),
        "rbf_sigma_frac": float(args.rbf_sigma_frac),
        "rbf_bias_init": float(args.rbf_bias_init),
        "rbf_weight_scale": float(args.rbf_weight_scale),
        "fractal_iter_max": int(max(1, args.fractal_iter_max)),
        "fractal_type": str(args.fractal_type).strip().lower(),
        "fractal_power": int(max(2, args.fractal_power)),
        "fractal_shift_frac": float(max(0.0, args.fractal_shift_frac)),
        "fractal_threshold_init": float(np.clip(args.fractal_threshold_init, -1.0, 1.0)),
        "fractal_center_origin": bool(args.fractal_center_origin),
        "branching_depth_max": int(max(1, args.branching_depth_max)),
        "branching_angle_frac": float(args.branching_angle_frac),
        "branching_length_decay": float(args.branching_length_decay),
        "branching_width_frac": float(args.branching_width_frac),
        "branching_width_decay": float(args.branching_width_decay),
        "branching_mirror_vertical": bool(args.branching_mirror_vertical),
        "branching_aim_electrodes": bool(args.branching_aim_electrodes),
        "branching_target_blend": float(args.branching_target_blend),
        "branching_root_x_frac": float(args.branching_root_x_frac),
        "branching_root_y_frac": float(args.branching_root_y_frac),
        "branching_seed_all_electrodes": bool(args.branching_seed_all_electrodes),
        "branching_force_meet_center": bool(args.branching_force_meet_center),
        "branching_meet_x_frac": float(args.branching_meet_x_frac),
        "branching_meet_y_frac": float(args.branching_meet_y_frac),
        "branching_meet_blend": float(args.branching_meet_blend),
        "branching_random_angle_frac": float(args.branching_random_angle_frac),
        "branching_random_center_boost": float(args.branching_random_center_boost),
        "branching_random_center_power": float(args.branching_random_center_power),
        "branching_force_touch_all_electrodes": bool(args.branching_force_touch_all_electrodes),
    }
    field_geometry = build_field_geometry(
        mesh_obj=mesh_obj,
        variable_mask=variable_mask,
        p1=p1,
        p2=p2,
        field_shape_cfg=field_shape_cfg,
    )
    field_shape_cfg["branching_electrode_points_norm"] = field_geometry.get("electrode_points_norm")
    field_shape_cfg["branching_electrode_points_norm_right"] = field_geometry.get("electrode_points_norm_right")
    if field_shape_cfg["model"] == "rbf":
        print(
            "Field shape model: "
            f"rbf_grid={field_shape_cfg['rbf_rows']}x{field_shape_cfg['rbf_cols']}, "
            f"sigma_frac={field_shape_cfg['rbf_sigma_frac']:.2f}"
        )
    else:
        fractal_label = field_shape_cfg["fractal_type"]
        if fractal_label == "multibrot":
            fractal_label = f"multibrot(power={field_shape_cfg['fractal_power']})"
        print(
            "Field shape model: "
            f"fractal ({fractal_label}), iter_max={field_shape_cfg['fractal_iter_max']}, "
            f"shift_frac={field_shape_cfg['fractal_shift_frac']:.2f}, "
            f"center_origin={field_shape_cfg['fractal_center_origin']}, "
            f"mirror_vertical={field_shape_cfg['branching_mirror_vertical']}, "
            f"aim_electrodes={field_shape_cfg['branching_aim_electrodes']}"
        )

    # Run optimizer
    if args.optimizer == "ga" and args.pop_size < 10:
        print(
            "Warning: pop_size < 10 can converge prematurely; "
            "consider >= 20 for better global search."
        )

    live_viewer = None
    if args.live_enabled:
        live_viewer = LiveMeshViewer(
            mesh_obj=mesh_obj,
            p1=p1,
            p2=p2,
            low_cond=args.low_cond,
            high_cond=args.high_cond,
            backend=args.live_backend,
            field_shape_cfg=field_shape_cfg,
        )

    if args.optimizer == "de":
        print(
            "Optimizer: differential evolution "
            f"(maxiter={args.de_maxiter}, popsize={args.de_popsize})"
        )
        result = optimize_parameterized_field_de(
            fwd=fwd,
            variable_mask=variable_mask,
            field_geometry=field_geometry,
            field_shape_cfg=field_shape_cfg,
            low_cond=args.low_cond,
            high_cond=args.high_cond,
            maxiter=args.de_maxiter,
            popsize=args.de_popsize,
            mutation=(args.de_mutation_min, args.de_mutation_max),
            recombination=args.de_recombination,
            seed=args.seed,
            adjacency=adjacency,
            electrode_supports=electrode_supports,
            generation_callback=(live_viewer.update if live_viewer is not None else None),
            live_update_every=args.live_update_every,
        )
    else:
        print(
            "Optimizer: genetic algorithm "
            f"(generations={args.generations}, pop_size={args.pop_size})"
        )
        result = optimize_parameterized_field_ga(
            fwd=fwd,
            variable_mask=variable_mask,
            field_geometry=field_geometry,
            field_shape_cfg=field_shape_cfg,
            low_cond=args.low_cond,
            high_cond=args.high_cond,
            generations=args.generations,
            pop_size=args.pop_size,
            elite_count=args.elite_count,
            crossover_rate=args.crossover_rate,
            mutation_rate=args.mutation_rate,
            init_high_fraction=args.init_high_fraction,
            tournament_size=args.tournament_size,
            seed=args.seed,
            adjacency=adjacency,
            electrode_supports=electrode_supports,
            generation_callback=(live_viewer.update if live_viewer is not None else None),
            live_update_every=args.live_update_every,
        )

    if live_viewer is not None:
        live_viewer.close()

    comparison_rows = []

    theta_ref = np.zeros(parameter_vector_size_from_cfg(field_shape_cfg), dtype=float)
    if field_shape_cfg["model"] == "rbf":
        theta_ref[0] = np.clip(field_shape_cfg["rbf_bias_init"], -1.0, 1.0)
    else:
        theta_ref[0] = np.clip(field_shape_cfg["fractal_threshold_init"], -1.0, 1.0)
    ref_state, ref_field = parameterized_state_from_theta(theta_ref, field_geometry, field_shape_cfg)
    ref_perm, ref_sens, ref_score = evaluate_state(
        fwd=fwd,
        state=ref_state,
        variable_mask=variable_mask,
        low_cond=args.low_cond,
        high_cond=args.high_cond,
        adjacency=adjacency,
        electrode_supports=electrode_supports,
    )
    comparison_rows.append(
        {
            "name": "Uniform Parameter Reference",
            "perm": ref_perm,
            "sensitivity": ref_sens,
            "score": float(ref_score),
        }
    )

    best_sens = np.asarray(result["best_sensitivity"]).ravel()
    print(f"Minimum entropy found: {result['best_score']:.6e}")
    print(f"Best sensitivity min: {best_sens.min():.6e}")
    print(f"Best sensitivity max: {best_sens.max():.6e}")
    print(f"Optimized field-shape parameters: {result['n_params']}")

    print(f"High elements in best state: {np.count_nonzero(result['best_state'])}")
    print(f"Low elements in best state: {result['n_variable'] - np.count_nonzero(result['best_state'])}")
    
    # Check connectivity of best state
    high_elems = result['best_state']
    is_connected = has_connected_high_elements(high_elems, adjacency)
    electrodes_linked = electrodes_connected_by_high_region(high_elems, adjacency, electrode_supports)
    high_components = get_connected_components(high_elems, adjacency)
    isolated_count = count_isolated_high_elements(high_elems, adjacency)
    print(f"High elements connected: {is_connected}")
    print(f"All electrodes connected by high region: {electrodes_linked}")
    print(f"High connected components: {len(high_components)}")
    print(f"Isolated high elements: {isolated_count}")

    plot_results(
        mesh_obj=mesh_obj,
        p1=p1,
        p2=p2,
        best_perm=result["best_perm"],
        best_sensitivity=result["best_sensitivity"],
        best_hist=result["best_history"],
        curr_hist=result["current_history"],
        best_score=result["best_score"],
        best_theta=result["best_theta"],
        field_shape_cfg=field_shape_cfg,
        comparison_rows=comparison_rows,
    )


def main():
    run()


if __name__ == "__main__":
    main()
