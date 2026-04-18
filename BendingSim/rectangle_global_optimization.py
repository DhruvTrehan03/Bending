from __future__ import absolute_import, division, print_function

import configparser
import numpy as np
import matplotlib.pyplot as plt
import os

import pyeit.eit.protocol as protocol
import pyeit.mesh as mesh
from pyeit.mesh.shape import rectangle
from pyeit.eit.fem import EITForward


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


def parameterized_state_from_theta(theta, field_geometry, field_shape_cfg):
    """Convert RBF parameters into a boolean high-conductivity state."""
    theta = np.asarray(theta, dtype=float).ravel()
    n_rows = int(field_shape_cfg["rbf_rows"])
    n_cols = int(field_shape_cfg["rbf_cols"])
    expected = parameter_vector_size(n_rows, n_cols)
    if theta.size != expected:
        raise ValueError(f"Expected {expected} parameters, got {theta.size}")

    bias = float(np.clip(theta[0], -1.0, 1.0))
    weights = np.clip(theta[1:], -1.0, 1.0)
    weight_scale = float(field_shape_cfg["rbf_weight_scale"])

    field = bias + weight_scale * field_geometry["basis"].dot(weights)
    state_high = field >= 0.0
    return np.asarray(state_high, dtype=bool), field


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
):
    """Optimize parameterized conductivity-field shape with a genetic algorithm."""
    rng = np.random.default_rng(seed)

    n_params = parameter_vector_size(int(field_shape_cfg["rbf_rows"]), int(field_shape_cfg["rbf_cols"]))
    generations = int(max(1, generations))
    pop_size = int(max(4, pop_size))
    elite_count = int(max(1, min(elite_count, pop_size - 1)))
    crossover_rate = float(min(max(crossover_rate, 0.0), 1.0))
    mutation_rate = float(min(max(mutation_rate, 0.0), 1.0))
    init_high_fraction = float(min(max(init_high_fraction, 0.0), 1.0))
    tournament_size = int(max(2, min(tournament_size, pop_size)))

    population = rng.normal(loc=0.0, scale=0.25, size=(pop_size, n_params))
    population[:, 0] = np.clip(
        rng.normal(loc=float(field_shape_cfg["rbf_bias_init"]), scale=0.20, size=pop_size),
        -1.0,
        1.0,
    )
    population[:, 1:] = np.clip(
        rng.normal(loc=0.0, scale=float(field_shape_cfg["rbf_weight_scale"]), size=(pop_size, n_params - 1)),
        -1.0,
        1.0,
    )

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

        if gen_best_score < best_score:
            best_score = gen_best_score
            best_theta = gen_best["theta"].copy()
            best_state = gen_best["state"].copy()
            best_thickness = gen_best["field"].copy()
            best_perm = gen_best["perm"].copy()
            best_sensitivity = gen_best["sensitivity"].copy()

        best_history.append(float(best_score))
        current_history.append(float(gen_best_score))

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


def plot_results(
    mesh_obj,
    p1,
    p2,
    best_perm,
    best_sensitivity,
    best_hist,
    curr_hist,
    best_score,
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
    
    # Initial conditions
    cfg.init_high_fraction = config.getfloat('initial_conditions', 'init_high_fraction', fallback=0.5)
    cfg.seed = config.getint('initial_conditions', 'seed', fallback=7)
    
    # Conductivity
    cfg.low_cond = config.getfloat('conductivity', 'low_cond', fallback=1e-6)
    cfg.high_cond = config.getfloat('conductivity', 'high_cond', fallback=1e6)
    
    # Mesh
    cfg.h0 = config.getfloat('mesh', 'h0', fallback=0.1)

    # Field-shape parameterization
    cfg.rbf_rows = config.getint('field_shape', 'rbf_rows', fallback=4)
    cfg.rbf_cols = config.getint('field_shape', 'rbf_cols', fallback=5)
    cfg.rbf_sigma_frac = config.getfloat('field_shape', 'rbf_sigma_frac', fallback=0.90)
    cfg.rbf_bias_init = config.getfloat('field_shape', 'rbf_bias_init', fallback=0.25)
    cfg.rbf_weight_scale = config.getfloat('field_shape', 'rbf_weight_scale', fallback=0.35)
    cfg.electrode_support_k = config.getint('field_shape', 'electrode_support_k', fallback=4)
    
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
        'rbf_rows': '4',
        'rbf_cols': '5',
        'rbf_sigma_frac': '0.90',
        'rbf_bias_init': '0.25',
        'rbf_weight_scale': '0.35',
        'electrode_support_k': '4',
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

    field_geometry = build_rbf_field_geometry(
        mesh_obj=mesh_obj,
        variable_mask=variable_mask,
        p1=p1,
        p2=p2,
        rbf_rows=args.rbf_rows,
        rbf_cols=args.rbf_cols,
        sigma_frac=args.rbf_sigma_frac,
    )
    field_shape_cfg = {
        "rbf_rows": int(max(1, args.rbf_rows)),
        "rbf_cols": int(max(1, args.rbf_cols)),
        "rbf_sigma_frac": float(args.rbf_sigma_frac),
        "rbf_bias_init": float(args.rbf_bias_init),
        "rbf_weight_scale": float(args.rbf_weight_scale),
    }
    print(
        "Field shape model: "
        f"rbf_grid={field_shape_cfg['rbf_rows']}x{field_shape_cfg['rbf_cols']}, "
        f"sigma_frac={field_shape_cfg['rbf_sigma_frac']:.2f}"
    )

    # Run genetic algorithm optimization
    if args.pop_size < 10:
        print(
            "Warning: pop_size < 10 can converge prematurely; "
            "consider >= 20 for better global search."
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
    )

    comparison_rows = []

    theta_ref = np.zeros(parameter_vector_size(field_shape_cfg["rbf_rows"], field_shape_cfg["rbf_cols"]), dtype=float)
    theta_ref[0] = np.clip(field_shape_cfg["rbf_bias_init"], -1.0, 1.0)
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
            "name": "Uniform RBF Reference",
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
        comparison_rows=comparison_rows,
    )


def main():
    run()


if __name__ == "__main__":
    main()
