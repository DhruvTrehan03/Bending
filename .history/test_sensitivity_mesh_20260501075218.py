"""Diagnostic script to test mesh and sensitivity visualization."""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from BendingSim.mesh_generation import build_model, visualize_mesh
from pyeit.eit.fem import EITForward
from pyeit.eit.jac import JAC

# Build test mesh
print("Building test mesh...")
mesh_obj, protocol_obj, variable_mask, p1, p2 = build_model(n_el=16, h0=0.1)

print(f"Mesh shape:")
print(f"  - Nodes: {mesh_obj.node.shape}")
print(f"  - Elements (triangles): {mesh_obj.element.shape}")
print(f"  - Electrodes: {mesh_obj.el_pos.shape}")
print(f"  - Variable mask: {variable_mask.shape}, non-zero: {np.sum(variable_mask)}")

# Setup FEM forward
print("\nSetting up FEM forward...")
fwd = EITForward(mesh_obj, protocol_obj)

# Create a simple test conductivity
n_elem = mesh_obj.elements
perm_uniform = np.ones(n_elem, dtype=float)  # All uniform

print(f"\nComputing Jacobian...")
jacobian, _ = fwd.compute_jac(perm=perm_uniform)
print(f"  - Jacobian shape: {jacobian.shape}")
print(f"  - Jacobian dtype: {jacobian.dtype}")

# Compute sensitivity (same as element_sensitivity_from_jacobian)
print(f"\nComputing sensitivity...")
sensitivity_raw = np.sum(np.abs(np.asarray(jacobian, dtype=float)), axis=0)
sensitivity_log = np.log10(sensitivity_raw + np.finfo(float).tiny)

print(f"  - Raw sensitivity shape: {sensitivity_raw.shape}")
print(f"  - Raw sensitivity range: [{np.min(sensitivity_raw):.6e}, {np.max(sensitivity_raw):.6e}]")
print(f"  - Raw sensitivity stats: mean={np.mean(sensitivity_raw):.6e}, std={np.std(sensitivity_raw):.6e}")
print(f"  - Log sensitivity shape: {sensitivity_log.shape}")
print(f"  - Log sensitivity range: [{np.min(sensitivity_log):.6e}, {np.max(sensitivity_log):.6e}]")
print(f"  - Log sensitivity stats: mean={np.mean(sensitivity_log):.6e}, std={np.std(sensitivity_log):.6e}")
print(f"  - Any NaN in log sensitivity: {np.any(np.isnan(sensitivity_log))}")
print(f"  - Any Inf in log sensitivity: {np.any(np.isinf(sensitivity_log))}")

# Check if sensitivity array size matches mesh elements
tri = mesh_obj.element
print(f"\nValidation:")
print(f"  - Triangle elements: {tri.shape[0]}")
print(f"  - Sensitivity per-element values: {sensitivity_log.size}")
print(f"  - Size match: {sensitivity_log.size == tri.shape[0]}")

# Visualize
print(f"\nVisualizing...")
pts = np.asarray(mesh_obj.node, dtype=float)
tri = np.asarray(mesh_obj.element, dtype=int)
el_pos = np.asarray(mesh_obj.el_pos, dtype=int).reshape(-1)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Plot 1: Mesh structure
ax = axes[0]
ax.triplot(pts[:, 0], pts[:, 1], tri, color="#748cab", linewidth=0.6, alpha=0.75)
if el_pos.size:
    ax.scatter(pts[el_pos, 0], pts[el_pos, 1], c="#111111", s=24, zorder=3)
ax.set_aspect("equal")
ax.set_title("Mesh Structure")
ax.grid(alpha=0.15)

# Plot 2: Raw sensitivity
ax = axes[1]
im2 = ax.tripcolor(pts[:, 0], pts[:, 1], tri, sensitivity_raw, shading="flat", cmap="viridis")
if el_pos.size:
    ax.scatter(pts[el_pos, 0], pts[el_pos, 1], c="#111111", s=24, zorder=3)
ax.set_aspect("equal")
ax.set_title("Raw Sensitivity")
fig.colorbar(im2, ax=ax, label="Sum |J[:, elem]|")

# Plot 3: Log sensitivity
ax = axes[2]
im3 = ax.tripcolor(pts[:, 0], pts[:, 1], tri, sensitivity_log, shading="flat", cmap="viridis")
if el_pos.size:
    ax.scatter(pts[el_pos, 0], pts[el_pos, 1], c="#111111", s=24, zorder=3)
ax.set_aspect("equal")
ax.set_title("Log10 Sensitivity")
fig.colorbar(im3, ax=ax, label="log10(sum |J[:, elem]|)")

plt.tight_layout()
plt.savefig('test_mesh_sensitivity.png', dpi=100, bbox_inches='tight')
print("\nPlot saved to test_mesh_sensitivity.png")

print("\nDone!")
