# =========================================================================
# EIT Jacobian Matrix - 16 Electrode Circular System
# Measurement Patterns: Adjacent-Adjacent (AD-AD) & Opposite-Adjacent (OP-AD)
# Python equivalent of EIDORS MATLAB workflow using pyEIT
# =========================================================================

from __future__ import absolute_import, division, print_function

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import pyeit.mesh as mesh
from pyeit.mesh.shape import circle
import pyeit.eit.protocol as protocol
from pyeit.eit.fem import EITForward

# -------------------------------------------------------------------------
# 1. CREATE 2D CIRCULAR MESH WITH 16 ELECTRODES
# -------------------------------------------------------------------------
n_el = 16          # number of electrodes
body_rad = 1.0     # domain radius (normalized)
mesh_maxh = 0.08   # max mesh element size (smaller = finer mesh)

mesh_obj = mesh.create(
    n_el=n_el,
    h0=mesh_maxh,
    fd=lambda pts: circle(pts, pc=[0, 0], r=body_rad),
    bbox=[[-body_rad, -body_rad], [body_rad, body_rad]],
)

pts = mesh_obj.node
tri = mesh_obj.element
el_pos = mesh_obj.el_pos
n_nodes = pts.shape[0]
n_elems = tri.shape[0]

print("Mesh created:")
print(f"  Nodes    : {n_nodes}")
print(f"  Elements : {n_elems}")
print(f"  Electrodes: {len(el_pos)}")

# -------------------------------------------------------------------------
# 2. DEFINE STIMULATION/MEASUREMENT PATTERNS
# -------------------------------------------------------------------------
# Adjacent-Adjacent (AD-AD): dist_exc=1, step_meas=1
proto_ad = protocol.create(n_el=n_el, dist_exc=1, step_meas=1, parser_meas="std")

# Opposite-Adjacent (OP-AD): dist_exc=n_el//2, step_meas=1
proto_op = protocol.create(n_el=n_el, dist_exc=n_el // 2, step_meas=1, parser_meas="std")

n_exc_ad = proto_ad.n_exc
n_meas_ad = proto_ad.n_meas_tot
n_exc_op = proto_op.n_exc
n_meas_op = proto_op.n_meas_tot

print(f"\nMeasurement patterns:")
print(f"  AD-AD: {n_exc_ad} injections -> {n_meas_ad} total measurements")
print(f"  OP-AD: {n_exc_op} injections -> {n_meas_op} total measurements")

# -------------------------------------------------------------------------
# 3. COMPUTE JACOBIAN MATRICES
# -------------------------------------------------------------------------
sigma_bg = 1.0  # background conductivity (S/m)
perm = np.full(n_elems, sigma_bg)

# -- AD-AD Jacobian --
fwd_ad = EITForward(mesh_obj, proto_ad)
J_ad, v0_ad = fwd_ad.compute_jac(perm=perm)

# -- OP-AD Jacobian --
fwd_op = EITForward(mesh_obj, proto_op)
J_op, v0_op = fwd_op.compute_jac(perm=perm)

print(f"\nJacobian sizes:")
print(f"  J_ad (AD-AD): {J_ad.shape[0]} measurements x {J_ad.shape[1]} elements")
print(f"  J_op (OP-AD): {J_op.shape[0]} measurements x {J_op.shape[1]} elements")

# -------------------------------------------------------------------------
# 4. MATRIX ANALYSIS
# -------------------------------------------------------------------------
sv_ad = la.svd(J_ad, compute_uv=False)
sv_op = la.svd(J_op, compute_uv=False)

cond_ad = sv_ad[0] / sv_ad[-1] if sv_ad[-1] != 0 else np.inf
cond_op = sv_op[0] / sv_op[-1] if sv_op[-1] != 0 else np.inf
rank_ad = np.linalg.matrix_rank(J_ad)
rank_op = np.linalg.matrix_rank(J_op)

print(f"\n--- Matrix Analysis ---")
print(f"AD-AD | Rank: {rank_ad}/{min(J_ad.shape)} | Condition number: {cond_ad:.3e}")
print(f"OP-AD | Rank: {rank_op}/{min(J_op.shape)} | Condition number: {cond_op:.3e}")

# -------------------------------------------------------------------------
# 5. FIGURE 1: Jacobian heatmaps + singular value spectra
# -------------------------------------------------------------------------
fig1, axes1 = plt.subplots(1, 3, figsize=(18, 5))
fig1.suptitle("EIT Jacobian Matrices", fontsize=14, fontweight="bold")

# AD-AD heatmap
im0 = axes1[0].imshow(J_ad, aspect="auto", cmap="jet")
axes1[0].set_title(f"Jacobian: AD-AD\n({J_ad.shape[0]} x {J_ad.shape[1]})",
                   fontsize=12, fontweight="bold")
axes1[0].set_xlabel("Element index")
axes1[0].set_ylabel("Measurement index")
plt.colorbar(im0, ax=axes1[0])

# OP-AD heatmap
im1 = axes1[1].imshow(J_op, aspect="auto", cmap="jet")
axes1[1].set_title(f"Jacobian: OP-AD\n({J_op.shape[0]} x {J_op.shape[1]})",
                   fontsize=12, fontweight="bold")
axes1[1].set_xlabel("Element index")
axes1[1].set_ylabel("Measurement index")
plt.colorbar(im1, ax=axes1[1])

# Singular value spectra
axes1[2].semilogy(sv_ad / sv_ad[0], "b-o", markersize=4, linewidth=1.2,
                  label=f"AD-AD (cond={cond_ad:.1e})")
axes1[2].semilogy(sv_op / sv_op[0], "r-s", markersize=4, linewidth=1.2,
                  label=f"OP-AD (cond={cond_op:.1e})")
axes1[2].set_xlabel("Index")
axes1[2].set_ylabel("Normalised singular value")
axes1[2].set_title("Singular Value Spectra", fontsize=12, fontweight="bold")
axes1[2].legend(loc="upper right")
axes1[2].grid(True)

plt.tight_layout()

# -------------------------------------------------------------------------
# 6. FIGURE 2: Sensitivity maps (aggregate |J| per element)
# -------------------------------------------------------------------------
sens_ad = np.sum(np.abs(J_ad), axis=0)  # sum across measurements
sens_op = np.sum(np.abs(J_op), axis=0)

fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
fig2.suptitle("Sensitivity Maps", fontsize=14, fontweight="bold")

xs, ys = pts[:, 0], pts[:, 1]

im2 = axes2[0].tripcolor(xs, ys, tri, sens_ad, shading="flat", cmap="jet")
axes2[0].plot(xs[el_pos], ys[el_pos], "ko", markersize=5)
axes2[0].set_aspect("equal")
axes2[0].set_title("Sensitivity Map: AD-AD", fontsize=12, fontweight="bold")
plt.colorbar(im2, ax=axes2[0])

im3 = axes2[1].tripcolor(xs, ys, tri, sens_op, shading="flat", cmap="jet")
axes2[1].plot(xs[el_pos], ys[el_pos], "ko", markersize=5)
axes2[1].set_aspect("equal")
axes2[1].set_title("Sensitivity Map: OP-AD", fontsize=12, fontweight="bold")
plt.colorbar(im3, ax=axes2[1])

plt.tight_layout()

# -------------------------------------------------------------------------
# 7. EXPORT JACOBIANS TO CSV
# -------------------------------------------------------------------------
np.savetxt("Jacobian_AD_AD.csv", J_ad, delimiter=",")
np.savetxt("Jacobian_OP_AD.csv", J_op, delimiter=",")
print(f"\nJacobians saved: Jacobian_AD_AD.csv and Jacobian_OP_AD.csv")
print("Done.")

plt.show()
