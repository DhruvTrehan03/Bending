# coding: utf-8
""" demo on forward 2D """
# Copyright (c) Benyuan Liu. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from pathlib import Path
import pyeit.eit.protocol as protocol
import pyeit.mesh as mesh
from pyeit.mesh.shape import rectangle
from pyeit.eit.fem import EITForward
from pyeit.eit.jac import JAC

def rect_electrode_positions(n_el, p1, p2):
    """
    Uniformly spaced electrodes (arc-length) around the rectangle boundary.
    Starts at offset = 0 on the bottom-left corner.
    Requires n_el % 4 == 0.
    """
    assert n_el % 4 == 0, "n_el must be divisible by 4"
    x1, y1 = p1
    x2, y2 = p2
    w, h = x2 - x1, y2 - y1
    perimeter = 2 * (w + h)
    spacing = perimeter / n_el
    offset = 0.0
    pts = []
    for i in range(n_el):
        s = offset + i * spacing
        if s < w:                   # bottom edge (left -> right)
            pts.append([x1 + s, y1])
        elif s < w + h:             # right edge (bottom -> top)
            pts.append([x2, y1 + (s - w)])
        elif s < 2 * w + h:        # top edge (right -> left)
            pts.append([x2 - (s - w - h), y2])
        else:                       # left edge (top -> bottom)
            pts.append([x1, y2 - (s - 2 * w - h)])
    return np.array(pts)


def rect_skip_electrode_positions(n_el, p1, p2, skip_x1, skip_x2):
    """
    Uniformly spaced electrodes on the boundary of p1→p2, skipping
    x=[skip_x1, skip_x2] on the top and bottom edges.
    The total path length equals the perimeter of a (w - skip_width) × h rectangle.
    """
    x1, y1 = p1
    x2, y2 = p2
    # Six segments traversed counter-clockwise
    segs = [
        ([x1,      y1], [skip_x1, y1]),   # bottom-left
        ([skip_x2, y1], [x2,      y1]),   # bottom-right
        ([x2,      y1], [x2,      y2]),   # right side
        ([x2,      y2], [skip_x2, y2]),   # top-right  (right → left)
        ([skip_x1, y2], [x1,      y2]),   # top-left   (right → left)
        ([x1,      y2], [x1,      y1]),   # left side
    ]
    lengths = [abs(e[0]-s[0]) + abs(e[1]-s[1]) for s, e in segs]
    total   = sum(lengths)
    spacing = total / n_el
    pts = []
    for i in range(n_el):
        s = i * spacing
        cumlen = 0
        for (start, end), length in zip(segs, lengths):
            if s < cumlen + length:
                t = (s - cumlen) / length
                pts.append([start[0] + t * (end[0] - start[0]),
                             start[1] + t * (end[1] - start[1])])
                break
            cumlen += length
    return np.array(pts)


""" 0. Build meshes """
n_el = 16

# --- Mesh A: plain 4×2 rectangle ---
p1_a, p2_a = [0.0, 0.0], [4.0, 2.0]
p_fix_a = rect_electrode_positions(n_el, p1_a, p2_a)
mesh_a = mesh.create(
    n_el, h0=0.1,
    fd=lambda pts: rectangle(pts, p1=p1_a, p2=p2_a),
    bbox=[p1_a, p2_a], p_fix=p_fix_a,
)

# --- Mesh B: 5×2 rectangle, electrodes skip x=[2, 3] on top/bottom ---
p1_b, p2_b   = [0.0, 0.0], [5.0, 2.0]
skip_x1, skip_x2 = 2.0, 3.0          # 1×2 inclusion x-span
p_fix_b = rect_skip_electrode_positions(n_el, p1_b, p2_b, skip_x1, skip_x2)
mesh_b = mesh.create(
    n_el, h0=0.1,
    fd=lambda pts: rectangle(pts, p1=p1_b, p2=p2_b),
    bbox=[p1_b, p2_b], p_fix=p_fix_b,
)

""" 1. Protocol """
proto_a = protocol.create(n_el, dist_exc=8, step_meas=1, parser_meas="rotate_meas")
proto_b = protocol.create(n_el, dist_exc=8, step_meas=1, parser_meas="rotate_meas")

""" 2. Permittivity arrays """
pts_a,  tri_a  = mesh_a.node, mesh_a.element
pts_b,  tri_b  = mesh_b.node, mesh_b.element
el_pos_a, el_pos_b = mesh_a.el_pos, mesh_b.el_pos

# Mesh A: uniform background
perm_a = np.full(tri_a.shape[0], 1.0)

# Mesh B: 1×2 inclusion (full height) with high or low conductivity
centroids_b = np.mean(pts_b[tri_b], axis=1)
in_incl = (centroids_b[:, 0] >= skip_x1) & (centroids_b[:, 0] <= skip_x2)
perm_b_inc = np.full(tri_b.shape[0], 1.0); perm_b_inc[in_incl] = 1e6
perm_b_dec = np.full(tri_b.shape[0], 1.0); perm_b_dec[in_incl] = 1e-6

scenarios = [
    ("4×2, no anomaly",          mesh_a, proto_a, pts_a, tri_a, el_pos_a, perm_a,    p1_a, p2_a, None),
    ("5×2, high conductivity",   mesh_b, proto_b, pts_b, tri_b, el_pos_b, perm_b_inc, p1_b, p2_b, in_incl),
    ("5×2, low conductivity",    mesh_b, proto_b, pts_b, tri_b, el_pos_b, perm_b_dec, p1_b, p2_b, in_incl),
]

""" 3. Compute Jacobians """
results = []
for name, mesh_s, proto_s, pts_s, tri_s, el_pos_s, perm_s, p1_s, p2_s, excl_mask_s in scenarios:
    fwd = EITForward(mesh_s, proto_s)
    J, v0 = fwd.compute_jac(perm=perm_s)
    sensitivity = np.log10(np.sum(J ** 2, axis=0) + np.finfo(float).tiny)
    results.append((name, pts_s, tri_s, el_pos_s, perm_s, sensitivity, v0, p1_s, p2_s, excl_mask_s))

""" 4. Plot: row1 = mesh, row2 = Jacobian sensitivity, row3 = base voltage, rows 4-6 = reconstructions """
all_perm = np.concatenate([r[4] for r in results])
perm_norm = mcolors.LogNorm(vmin=all_perm.min(), vmax=all_perm.max())

# Circle anomaly positions for reconstruction:
#   - Left: no overlap with middle region (x=2-3)
#   - Overlap: overlapping with edge of middle region
#   - Center: in the center of the middle region
anomaly_radius = 0.3
anomaly_delta = 9.0  # Absolute change: anomaly = background + delta

# Anomaly positions relative to each mesh's geometry
circle_anomalies = [
    ("Left", 1.0),       # Left side, no overlap with skip region
    ("Overlap", 2.0),    # Overlapping with skip region boundary  
    ("Center", 2.5),     # Center of the skip region
]

row_figs_axes = [plt.subplots(1, 3, figsize=(18, 4.2)) for _ in range(9)]
jac_cmaps = [plt.cm.viridis, plt.cm.viridis, plt.cm.viridis]
row_labels = [
    "mesh",
    "jacobian",
    "base_voltage",
    "recon_left",
    "recon_overlap",
    "recon_center",
    "model4x2_left",
    "model4x2_overlap",
    "model4x2_center",
]

# Pre-build JAC solver for mesh_a (4×2) to use for "wrong model" reconstruction
fwd_4x2 = EITForward(mesh_a, proto_a)
jac_solver_4x2 = JAC(mesh_a, proto_a)
jac_solver_4x2.setup(p=0.20, lamb=0.001, method="kotre")
v0_ref_4x2 = fwd_4x2.solve_eit(perm=perm_a)

for col, (name, pts_s, tri_s, el_pos_s, perm_s, sensitivity, v0_s, p1_s, p2_s, excl_mask_s) in enumerate(results):
    xs, ys = pts_s[:, 0], pts_s[:, 1]
    ylim = [p1_s[1] - 0.2, p2_s[1] + 0.2]
    xlim = [p1_s[0] - 0.2, p2_s[0] + 0.2]
    cy = (p1_s[1] + p2_s[1]) / 2.0  # Center y for this mesh

    # --- Row 0: mesh conductivity ---
    ax_m = row_figs_axes[0][1][col]
    im_m = ax_m.tripcolor(xs, ys, tri_s, perm_s, shading="flat",
                          cmap=plt.cm.RdBu_r, norm=perm_norm)
    ax_m.plot(xs[el_pos_s], ys[el_pos_s], "ko", markersize=4)
    ax_m.set_aspect("equal"); ax_m.set_ylim(ylim); ax_m.set_xlim(xlim)
    ax_m.set_title(f"Mesh — {name}")
    plt.colorbar(im_m, ax=ax_m, label="Conductivity (log scale)")

    # --- Row 1: Jacobian sensitivity ---
    ax_j = row_figs_axes[1][1][col]
    if excl_mask_s is not None:
        scale_vals = sensitivity[~excl_mask_s]
        scale_vals = scale_vals[np.isfinite(scale_vals)]
        if scale_vals.size > 0:
            jac_norm = mcolors.Normalize(vmin=scale_vals.min(), vmax=scale_vals.max())
        else:
            jac_norm = None
        sensitivity_plot = np.ma.array(sensitivity, mask=excl_mask_s)
    else:
        scale_vals = sensitivity[np.isfinite(sensitivity)]
        if scale_vals.size > 0:
            jac_norm = mcolors.Normalize(vmin=scale_vals.min(), vmax=scale_vals.max())
        else:
            jac_norm = None
        sensitivity_plot = sensitivity

    im_j = ax_j.tripcolor(xs, ys, tri_s, sensitivity_plot, shading="flat",
                          cmap=jac_cmaps[col], norm=jac_norm)
    ax_j.set_aspect("equal"); ax_j.set_ylim(ylim); ax_j.set_xlim(xlim)
    ax_j.set_title(f"Jacobian — {name}")
    plt.colorbar(im_j, ax=ax_j, label="log₁₀(sensitivity)")

    # --- Row 2: base voltage signal ---
    ax_v = row_figs_axes[2][1][col]
    v0_plot = np.abs(np.asarray(v0_s)).ravel()
    ax_v.plot(np.arange(v0_plot.size), v0_plot, "-", linewidth=1.5)
    ax_v.set_title(f"Base Voltage — {name}")
    ax_v.set_xlabel("Measurement index")
    ax_v.set_ylabel("|v₀|")
    ax_v.grid(True, alpha=0.3)

    # --- Rows 3-5: Reconstruction for each anomaly position ---
    # Get the mesh and protocol for this scenario
    mesh_s = scenarios[col][1]
    proto_s = scenarios[col][2]
    
    # Build JAC solver and forward model for this scenario
    fwd_s = EITForward(mesh_s, proto_s)
    jac_solver = JAC(mesh_s, proto_s)
    jac_solver.setup(p=0.20, lamb=0.001, method="kotre")
    
    # Reference measurement using scenario's background conductivity
    v0_ref = fwd_s.solve_eit(perm=perm_s)
    
    centroids_s = np.mean(pts_s[tri_s], axis=1)
    theta = np.linspace(0, 2*np.pi, 100)
    
    for row_offset, (anom_name, cx) in enumerate(circle_anomalies):
        ax_rec = row_figs_axes[3 + row_offset][1][col]
        
        # Create permittivity with circular anomaly
        dist_from_center = np.sqrt((centroids_s[:, 0] - cx)**2 + (centroids_s[:, 1] - cy)**2)
        in_circle = dist_from_center <= anomaly_radius
        
        # Apply absolute change (add delta to background conductivity)
        perm_anom = perm_s.copy()
        perm_anom[in_circle] = perm_s[in_circle] + anomaly_delta
        
        # Forward solve with anomaly
        v1 = fwd_s.solve_eit(perm=perm_anom)
        
        # Reconstruct
        ds = jac_solver.solve(v1, v0_ref, normalize=True)
        ds_plot = np.real(ds)
        
        im_rec = ax_rec.tripcolor(xs, ys, tri_s, ds_plot, shading="flat",
                                   cmap=plt.cm.RdBu_r)
        ax_rec.plot(xs[el_pos_s], ys[el_pos_s], "ko", markersize=4)
        # Draw circle outline showing true anomaly location
        ax_rec.plot(cx + anomaly_radius*np.cos(theta), cy + anomaly_radius*np.sin(theta), 
                    'g-', linewidth=2)
        ax_rec.set_aspect("equal"); ax_rec.set_ylim(ylim); ax_rec.set_xlim(xlim)
        ax_rec.set_title(f"Recon {anom_name} — {name}")
        plt.colorbar(im_rec, ax=ax_rec, label="Δσ (normalized)")

    # --- Rows 6-8: Reconstruction assuming 4×2 mesh for all scenarios (one per anomaly) ---
    cy_4x2 = 1.0  # Center y for 4×2 mesh
    xs_4x2, ys_4x2 = pts_a[:, 0], pts_a[:, 1]
    
    for row_offset, (anom_name, cx_4x2) in enumerate(circle_anomalies):
        ax_4x2 = row_figs_axes[6 + row_offset][1][col]
        
        # Create anomaly on the actual mesh
        dist_from_center = np.sqrt((centroids_s[:, 0] - cx_4x2)**2 + (centroids_s[:, 1] - cy_4x2)**2)
        in_circle = dist_from_center <= anomaly_radius
        
        # Apply absolute change (add delta to background conductivity)
        perm_anom_4x2 = perm_s.copy()
        perm_anom_4x2[in_circle] = perm_s[in_circle] + anomaly_delta
        
        # Forward solve on actual mesh
        v1_actual = fwd_s.solve_eit(perm=perm_anom_4x2)
        
        # Reconstruct using the 4×2 mesh (possibly wrong model)
        ds_4x2 = jac_solver_4x2.solve(v1_actual, v0_ref_4x2, normalize=True)
        ds_4x2_plot = np.real(ds_4x2)
        
        # Plot on the 4×2 mesh
        im_4x2 = ax_4x2.tripcolor(xs_4x2, ys_4x2, tri_a, ds_4x2_plot, shading="flat",
                                   cmap=plt.cm.RdBu_r)
        ax_4x2.plot(xs_4x2[el_pos_a], ys_4x2[el_pos_a], "ko", markersize=4)
        # Draw circle outline showing true anomaly location
        ax_4x2.plot(cx_4x2 + anomaly_radius*np.cos(theta), cy_4x2 + anomaly_radius*np.sin(theta), 
                    'g-', linewidth=2)
        ax_4x2.set_aspect("equal")
        ax_4x2.set_ylim([p1_a[1] - 0.2, p2_a[1] + 0.2])
        ax_4x2.set_xlim([p1_a[0] - 0.2, p2_a[0] + 0.2])
        ax_4x2.set_title(f"4×2 Model {anom_name} — {name}")
        plt.colorbar(im_4x2, ax=ax_4x2, label="Δσ (normalized)")

output_dir = Path(__file__).resolve().parent / "rectangleBend_pngs"
output_dir.mkdir(parents=True, exist_ok=True)

for row_idx, (fig, _) in enumerate(row_figs_axes, start=1):
    fig.tight_layout()
    fig.savefig(output_dir / f"row_{row_idx:02d}_{row_labels[row_idx - 1]}.png", dpi=300)
plt.show()