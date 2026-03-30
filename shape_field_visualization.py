# coding: utf-8
"""
Visualize electric field for circular EIT setup with different shapes in center
Shapes: Square, Circle, Triangle, Star, Plus
"""
from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import numpy as np
import pyeit.eit.protocol as protocol
import pyeit.mesh as mesh
import pyeit.eit.jac as jac
from pyeit.eit.fem import Forward, EITForward
from pyeit.mesh.wrapper import PyEITAnomaly_Circle, PyEITAnomaly, PyEITMesh
from pyeit.eit.interp2d import sim2pts, pdegrad
from dataclasses import dataclass
from typing import Any


# Custom Shape Anomaly Classes
@dataclass
class PyEITAnomaly_Square(PyEITAnomaly):
    """Square anomaly centered at given position"""
    r: float = 0.2  # half-width of the square
    
    def mask(self, pts: np.ndarray) -> Any:
        """Return boolean mask for square shape"""
        pts = pts[:, :2].reshape((-1, 2))
        x_diff = np.abs(pts[:, 0] - self.center[0])
        y_diff = np.abs(pts[:, 1] - self.center[1])
        return (x_diff <= self.r) & (y_diff <= self.r)


@dataclass
class PyEITAnomaly_Triangle(PyEITAnomaly):
    """Equilateral triangle anomaly centered at given position"""
    r: float = 0.25  # radius of circumscribed circle
    
    def mask(self, pts: np.ndarray) -> Any:
        """Return boolean mask for equilateral triangle shape"""
        pts = pts[:, :2].reshape((-1, 2))
        x = pts[:, 0] - self.center[0]
        y = pts[:, 1] - self.center[1]
        
        # Equilateral triangle vertices (pointing up)
        height = self.r * 3 / 2
        v1_y = height * 2/3  # top vertex
        v2_y = -height * 1/3  # bottom vertices
        v_x = self.r * np.sqrt(3) / 2
        
        # Check if point is inside triangle using cross products
        # Edge 1: from bottom-left to top
        edge1 = (x + v_x) * (v1_y - v2_y) - (y - v2_y) * (v_x)
        # Edge 2: from top to bottom-right
        edge2 = (x - 0) * (v2_y - v1_y) - (y - v1_y) * (v_x)
        # Edge 3: from bottom-right to bottom-left
        edge3 = (x - v_x) * (v2_y - v2_y) - (y - v2_y) * (-2 * v_x)
        
        return (edge1 >= 0) & (edge2 >= 0) & (edge3 >= 0)


@dataclass
class PyEITAnomaly_Star(PyEITAnomaly):
    """5-pointed star anomaly centered at given position"""
    r_outer: float = 0.25  # outer radius
    r_inner: float = 0.1   # inner radius
    
    def mask(self, pts: np.ndarray) -> Any:
        """Return boolean mask for star shape"""
        pts = pts[:, :2].reshape((-1, 2))
        x = pts[:, 0] - self.center[0]
        y = pts[:, 1] - self.center[1]
        
        # Convert to polar coordinates
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        
        # 5-pointed star: modulate radius based on angle
        n_points = 5
        angle_per_point = 2 * np.pi / n_points
        
        # Normalize angle to [0, angle_per_point]
        local_angle = np.mod(theta + np.pi/2, angle_per_point)
        
        # Linear interpolation between inner and outer radius
        # Maximum radius at 0, minimum at angle_per_point/2
        radius_at_angle = self.r_inner + (self.r_outer - self.r_inner) * \
                         (1 - 2 * np.abs(local_angle - angle_per_point/2) / angle_per_point)
        
        return r <= radius_at_angle


@dataclass
class PyEITAnomaly_Plus(PyEITAnomaly):
    """Plus/cross anomaly centered at given position"""
    length: float = 0.4  # length of each arm
    width: float = 0.12   # width of each arm
    
    def mask(self, pts: np.ndarray) -> Any:
        """Return boolean mask for plus/cross shape"""
        pts = pts[:, :2].reshape((-1, 2))
        x = np.abs(pts[:, 0] - self.center[0])
        y = np.abs(pts[:, 1] - self.center[1])
        
        # Horizontal arm
        horizontal = (x <= self.length/2) & (y <= self.width/2)
        # Vertical arm
        vertical = (y <= self.length/2) & (x <= self.width/2)
        
        return horizontal | vertical


def set_perm_with_mask(mesh_obj, anomaly, background=1.0):
    """Set element permittivity using anomaly masks.

    This works around a pyEIT issue where set_perm can ignore masks.
    """
    perm = np.full(mesh_obj.n_elems, background, dtype=float)
    tri_centers = mesh_obj.elem_centers

    anomalies = anomaly if isinstance(anomaly, (list, tuple)) else [anomaly]
    for an in anomalies:
        mask = an.mask(tri_centers)
        perm[mask] = np.real(an.perm)

    return PyEITMesh(
        node=mesh_obj.node,
        element=mesh_obj.element,
        perm=perm,
        el_pos=mesh_obj.el_pos,
        ref_node=mesh_obj.ref_node,
    )


def compute_field_for_shape(mesh_obj, anomaly, ex_line):
    """Compute electric field for a given anomaly shape"""
    # Set permittivity with anomaly
    mesh_new = set_perm_with_mask(mesh_obj, anomaly=anomaly, background=1.0)
    
    # Solve forward problem
    fwd = Forward(mesh_new)
    f = fwd.solve(ex_line)
    f = np.real(f)
    
    # Compute field gradient
    pts = mesh_obj.node
    tri = mesh_obj.element
    ux, uy = pdegrad(pts, tri, f)
    uf = ux**2 + uy**2
    uf_pts = sim2pts(pts, tri, uf)
    uf_logpwr = 10 * np.log10(uf_pts + 1e-10)  # Add small value to avoid log(0)
    
    return f, uf_logpwr, mesh_new.perm


def compute_reconstruction(mesh_obj, protocol_obj, anomaly):
    """Compute JAC difference-imaging reconstruction for an anomaly."""
    mesh_ref = PyEITMesh(
        node=mesh_obj.node,
        element=mesh_obj.element,
        perm=np.ones(mesh_obj.n_elems, dtype=float),
        el_pos=mesh_obj.el_pos,
        ref_node=mesh_obj.ref_node,
    )
    mesh_new = set_perm_with_mask(mesh_obj, anomaly=anomaly, background=1.0)

    fwd_ref = EITForward(mesh_ref, protocol_obj)
    v0 = np.real(fwd_ref.solve_eit())

    fwd_new = EITForward(mesh_new, protocol_obj)
    v1 = np.real(fwd_new.solve_eit())

    eit_solver = jac.JAC(mesh_ref, protocol_obj)
    eit_solver.setup(p=0.5, lamb=0.005, method="kotre", jac_normalized=True)
    ds = np.real(eit_solver.solve(v1, v0, normalize=True))

    return sim2pts(mesh_obj.node, mesh_obj.element, ds)


def main():
    """Main function to create and visualize fields for different shapes"""
    
    # Build circular mesh
    print("Building mesh...")
    n_el = 16  # number of electrodes
    mesh_obj = mesh.create(n_el, h0=0.05)
    
    # Extract mesh information
    pts = mesh_obj.node
    tri = mesh_obj.element
    el_pos = mesh_obj.el_pos
    x, y = pts[:, 0], pts[:, 1]
    
    mesh_obj.print_stats()
    
    # Setup EIT scan protocol
    protocol_obj = protocol.create(n_el, dist_exc=1, step_meas=1, parser_meas="std")
    ex_line = protocol_obj.ex_mat[0].ravel()
    
    # Define shapes with their anomalies (decreased conductivity: perm=0.1)
    shapes = {
        'Circle': PyEITAnomaly_Circle(center=[0, 0], r=0.35, perm=0.1),
        'Square': PyEITAnomaly_Square(center=[0, 0], r=0.35, perm=0.1),
        'Triangle': PyEITAnomaly_Triangle(center=[0, 0], r=0.45, perm=0.1),
        'Star': PyEITAnomaly_Star(center=[0, 0], r_outer=0.4, r_inner=0.18, perm=0.1),
        'Plus': PyEITAnomaly_Plus(center=[0, 0], length=0.65, width=0.2, perm=0.1)
    }
    
    # Create figure with subplots for equipotential lines
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    print("\nComputing fields for different shapes...")
    
    for idx, (shape_name, anomaly) in enumerate(shapes.items()):
        print(f"  Processing {shape_name}...")
        
        # Compute field
        f, uf_logpwr, perm = compute_field_for_shape(mesh_obj, anomaly, ex_line)
        
        # Plot equipotential lines
        ax = axes[idx]
        
        # Draw contour lines (equipotential)
        vf = np.linspace(min(f), max(f), 20)
        ax.tricontour(x, y, tri, f, vf, cmap=plt.cm.viridis, linewidths=1.5)
        
        # Draw the anomaly region
        ax.tripcolor(
            x, y, tri,
            np.real(perm),
            edgecolors="k",
            shading="flat",
            alpha=0.3,
            cmap=plt.cm.Blues,
            vmin=0.1, vmax=1.0
        )

        # Draw a clear anomaly boundary so each shape is visible
        perm_nodes = sim2pts(pts, tri, np.real(perm))
        ax.tricontour(x, y, tri, perm_nodes, levels=[0.55], colors='magenta', linewidths=2.0)
        
        # Draw electrodes
        ax.plot(x[el_pos], y[el_pos], 'ko', markersize=8, markerfacecolor='red')
        
        # Formatting
        ax.set_aspect("equal")
        ax.set_xlim([-1.2, 1.2])
        ax.set_ylim([-1.2, 1.2])
        ax.set_title(f'{shape_name} - Equipotential Lines', fontsize=14, fontweight='bold')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True, alpha=0.3)
    
    # Plot E-field magnitude for one example (Circle) in the last subplot
    ax = axes[5]
    f, uf_logpwr, perm = compute_field_for_shape(mesh_obj, shapes['Circle'], ex_line)
    
    im = ax.tripcolor(x, y, tri, uf_logpwr, cmap=plt.cm.hot, shading='flat')
    ax.tricontour(x, y, tri, uf_logpwr, 10, colors='cyan', linewidths=0.5, alpha=0.7)
    ax.plot(x[el_pos], y[el_pos], 'ko', markersize=8, markerfacecolor='lime')
    
    ax.set_aspect("equal")
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_title('Circle - E-field Magnitude (dB)', fontsize=14, fontweight='bold')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Log E-field', rotation=270, labelpad=20)
    
    plt.tight_layout()
    
    # Create a second figure showing voltage distribution for all shapes
    fig2, axes2 = plt.subplots(2, 3, figsize=(18, 12))
    axes2 = axes2.flatten()
    
    print("\nPlotting voltage distributions...")
    
    for idx, (shape_name, anomaly) in enumerate(shapes.items()):
        print(f"  Plotting {shape_name} voltage...")
        
        # Compute field
        f, uf_logpwr, perm = compute_field_for_shape(mesh_obj, anomaly, ex_line)
        
        # Plot voltage distribution
        ax = axes2[idx]
        im = ax.tripcolor(x, y, tri, f, cmap=plt.cm.viridis, shading='flat')
        ax.tricontour(x, y, tri, f, 15, colors='white', linewidths=0.5, alpha=0.5)
        
        # Draw the anomaly region as filled color
        perm_nodes = sim2pts(pts, tri, np.real(perm))
        ax.tricontour(x, y, tri, perm_nodes, levels=[0.5], colors='red', linewidths=2)
        
        ax.plot(x[el_pos], y[el_pos], 'ro', markersize=8)
        
        ax.set_aspect("equal")
        ax.set_xlim([-1.2, 1.2])
        ax.set_ylim([-1.2, 1.2])
        ax.set_title(f'{shape_name} - Voltage Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Voltage (V)', rotation=270, labelpad=20)
    
    # Hide the last subplot
    axes2[5].axis('off')
    
    plt.tight_layout()
    
    # Create a third figure showing E-field for all shapes
    fig3, axes3 = plt.subplots(2, 3, figsize=(18, 12))
    axes3 = axes3.flatten()
    
    print("\nPlotting E-field magnitudes...")
    
    for idx, (shape_name, anomaly) in enumerate(shapes.items()):
        print(f"  Plotting {shape_name} E-field...")
        
        # Compute field
        f, uf_logpwr, perm = compute_field_for_shape(mesh_obj, anomaly, ex_line)
        
        # Plot E-field magnitude
        ax = axes3[idx]
        im = ax.tripcolor(x, y, tri, uf_logpwr, cmap=plt.cm.hot, shading='flat')
        ax.tricontour(x, y, tri, uf_logpwr, 10, colors='cyan', linewidths=0.5, alpha=0.7)
        ax.plot(x[el_pos], y[el_pos], 'ko', markersize=8, markerfacecolor='lime')
        
        ax.set_aspect("equal")
        ax.set_xlim([-1.2, 1.2])
        ax.set_ylim([-1.2, 1.2])
        ax.set_title(f'{shape_name} - E-field Magnitude (dB)', fontsize=14, fontweight='bold')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Log E-field', rotation=270, labelpad=20)
    
    # Hide the last subplot
    axes3[5].axis('off')
    
    plt.tight_layout()
    
    print("\n✓ Field visualizations complete!")
    print("  Figure 1: Equipotential lines for all shapes")
    print("  Figure 2: Voltage distributions for all shapes")
    print("  Figure 3: E-field magnitudes for all shapes")

    # Create a fourth figure showing JAC reconstructions for all shapes
    fig4, axes4 = plt.subplots(2, 3, figsize=(18, 12))
    axes4 = axes4.flatten()

    print("\nComputing JAC reconstructions...")

    for idx, (shape_name, anomaly) in enumerate(shapes.items()):
        print(f"  Reconstructing {shape_name}...")

        ds_nodes = compute_reconstruction(mesh_obj, protocol_obj, anomaly)

        ax = axes4[idx]
        vmax = np.nanmax(np.abs(ds_nodes))
        if not np.isfinite(vmax) or vmax == 0:
            vmax = 1.0

        im = ax.tripcolor(
            x,
            y,
            tri,
            ds_nodes,
            cmap=plt.cm.RdBu_r,
            shading='flat',
            vmin=-vmax,
            vmax=vmax,
        )
        ax.plot(x[el_pos], y[el_pos], 'ko', markersize=7, markerfacecolor='yellow')

        ax.set_aspect("equal")
        ax.set_xlim([-1.2, 1.2])
        ax.set_ylim([-1.2, 1.2])
        ax.set_title(f'{shape_name} - JAC Reconstruction', fontsize=14, fontweight='bold')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True, alpha=0.3)

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Δ Conductivity (a.u.)', rotation=270, labelpad=20)

    # Hide the last subplot
    axes4[5].axis('off')

    plt.tight_layout()
    print("  Figure 4: JAC reconstructions for all shapes")
    
    plt.show()


if __name__ == "__main__":
    main()
