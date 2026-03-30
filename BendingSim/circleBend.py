# coding: utf-8
""" demo on forward 2D (circle domain with rectangular middle inclusion) """
# Copyright (c) Benyuan Liu. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.tri as mtri
import numpy as np
import pyeit.eit.protocol as protocol
import pyeit.mesh as mesh
from pyeit.mesh.shape import circle
from pyeit.eit.fem import EITForward

""" 0. Build meshes """
n_el = 16

# --- Mesh A: plain circle ---
mesh_a = mesh.create(
    n_el,
    h0=0.02,
    fd=circle,
    bbox=[[-1.0, -1.0], [1.0, 1.0]],
)

# --- Mesh B: circle with rectangular middle inclusion ---
mesh_b = mesh.create(
    n_el,
    h0=0.02,
    fd=circle,
    bbox=[[-1.0, -1.0], [1.0, 1.0]],
)

""" 1. Protocol """
proto_a = protocol.create(n_el, dist_exc=8, step_meas=1, parser_meas="rotate_meas")
proto_b = protocol.create(n_el, dist_exc=8, step_meas=1, parser_meas="rotate_meas")

""" 2. Permittivity arrays """
pts_a, tri_a = mesh_a.node, mesh_a.element
pts_b, tri_b = mesh_b.node, mesh_b.element
el_pos_a, el_pos_b = mesh_a.el_pos, mesh_b.el_pos

# Mesh A: uniform background
perm_a = np.full(tri_a.shape[0], 1.0)

# Mesh B: rectangular middle inclusion with high or low conductivity
# Rectangle passes through the gap between electrodes 0 and 1 (and opposite gap)
# at an angle, not touching any electrode.
el_coords_b = pts_b[el_pos_b]
# Electrodes 0 and 1 angles
theta0 = np.arctan2(el_coords_b[0, 1], el_coords_b[0, 0])
theta1 = np.arctan2(el_coords_b[1, 1], el_coords_b[1, 0])
# Midpoint angle (gap direction)
gap_angle = (theta0 + theta1) / 2

# Rectangle half-width (perpendicular to gap direction) and half-length (along gap)
rect_half_width = 0.1   # narrow strip
rect_half_length = 1  # extends nearly to boundary but doesn't touch electrodes

centroids_b = np.mean(pts_b[tri_b], axis=1)
# Rotate centroids by -gap_angle to align rectangle with axes
cos_a, sin_a = np.cos(-gap_angle), np.sin(-gap_angle)
cx_rot = centroids_b[:, 0] * cos_a - centroids_b[:, 1] * sin_a
cy_rot = centroids_b[:, 0] * sin_a + centroids_b[:, 1] * cos_a
# Check if rotated centroids fall within axis-aligned rectangle
in_incl = (
    (np.abs(cx_rot) <= rect_half_length)
    & (np.abs(cy_rot) <= rect_half_width)
)
perm_b_inc = np.full(tri_b.shape[0], 1.0)
perm_b_inc[in_incl] = 1e6
perm_b_dec = np.full(tri_b.shape[0], 1.0)
perm_b_dec[in_incl] = 1e-6

scenarios = [
    ("Circle, no anomaly", mesh_a, proto_a, pts_a, tri_a, el_pos_a, perm_a, None),
    ("Circle, high conductivity", mesh_b, proto_b, pts_b, tri_b, el_pos_b, perm_b_inc, in_incl),
    ("Circle, low conductivity", mesh_b, proto_b, pts_b, tri_b, el_pos_b, perm_b_dec, in_incl),
]

""" 3. Compute Jacobians """
results = []
for name, mesh_s, proto_s, pts_s, tri_s, el_pos_s, perm_s, excl_mask_s in scenarios:
    fwd = EITForward(mesh_s, proto_s)
    J, v0 = fwd.compute_jac(perm=perm_s)
    sensitivity = np.log10(np.sum(J ** 2, axis=0) + np.finfo(float).tiny)
    results.append((name, pts_s, tri_s, el_pos_s, perm_s, sensitivity, v0, excl_mask_s))

""" 4. Plot: row1 = mesh, row2 = Jacobian sensitivity, row3 = base voltage """
all_perm = np.concatenate([r[4] for r in results])
perm_norm = mcolors.LogNorm(vmin=all_perm.min(), vmax=all_perm.max())

fig, axes = plt.subplots(3, 3, figsize=(18, 13))
jac_cmaps = [plt.cm.viridis, plt.cm.viridis, plt.cm.viridis]

for col, (name, pts_s, tri_s, el_pos_s, perm_s, sensitivity, v0_s, excl_mask_s) in enumerate(results):
    xs, ys = pts_s[:, 0], pts_s[:, 1]
    ylim = [-1.1, 1.1]
    xlim = [-1.1, 1.1]

    # --- Top row: mesh conductivity ---
    ax_m = axes[0, col]
    im_m = ax_m.tripcolor(xs, ys, tri_s, perm_s, shading="flat", cmap=plt.cm.RdBu_r, norm=perm_norm)
    ax_m.plot(xs[el_pos_s], ys[el_pos_s], "ko", markersize=4)
    ax_m.set_aspect("equal")
    ax_m.set_ylim(ylim)
    ax_m.set_xlim(xlim)
    ax_m.set_title(f"Mesh — {name}")
    plt.colorbar(im_m, ax=ax_m, label="Conductivity (log scale)")

    # --- Middle row: Jacobian sensitivity ---
    ax_j = axes[1, col]
    if excl_mask_s is not None:
        # Exclude rectangle region from colormap normalization only
        scale_vals = sensitivity[~excl_mask_s]
        scale_vals = scale_vals[np.isfinite(scale_vals)]
        if scale_vals.size > 0:
            jac_norm = mcolors.Normalize(vmin=scale_vals.min(), vmax=scale_vals.max())
        else:
            jac_norm = None
    else:
        scale_vals = sensitivity[np.isfinite(sensitivity)]
        if scale_vals.size > 0:
            jac_norm = mcolors.Normalize(vmin=scale_vals.min(), vmax=scale_vals.max())
        else:
            jac_norm = None

    # Plot full Jacobian (including rectangle region) but with normalization from non-rectangle only
    im_j = ax_j.tripcolor(xs, ys, tri_s, sensitivity, shading="flat", cmap=jac_cmaps[col], norm=jac_norm)
    ax_j.set_aspect("equal")
    ax_j.set_ylim(ylim)
    ax_j.set_xlim(xlim)
    ax_j.set_title(f"Jacobian — {name}")
    plt.colorbar(im_j, ax=ax_j, label="log₁₀(sensitivity)")

    # --- Bottom row: base voltage signal ---
    ax_v = axes[2, col]
    v0_plot = np.abs(np.asarray(v0_s)).ravel()
    ax_v.plot(np.arange(v0_plot.size), v0_plot, "-", linewidth=1.5)
    ax_v.set_title(f"Base Voltage — {name}")
    ax_v.set_xlabel("Measurement index")
    ax_v.set_ylabel("|v₀|")
    ax_v.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
