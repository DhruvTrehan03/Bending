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


def make_permittivity(state_high, variable_mask, low_cond, high_cond):
    """Construct element conductivity from a boolean high/low element state."""
    n_elem = variable_mask.size
    perm = np.full(n_elem, low_cond, dtype=float)

    variable_indices = np.where(variable_mask)[0]
    perm[variable_indices[state_high]] = high_cond
    return perm


def element_sensitivity_from_jacobian(jacobian):
    """Calculate element sensitivity as the L1 norm of jacobian columns.
    
    Returns sensitivity values without log scaling to preserve dynamic range information
    while avoiding extreme values.
    """
    # Use L1 norm without log to avoid extreme values
    sensitivity = np.sum(np.abs(jacobian), axis=0)
    # Add small epsilon to avoid zero sensitivity
    return sensitivity + np.finfo(float).tiny


def uniformity_score(sensitivity):
    """Compute a score that penalizes non-uniform sensitivity distribution.
    
    Lower score = more uniform distribution with no extremely low values.
    Uses coefficient of variation (std/mean) to measure uniformity,
    penalizes minimum values that are too low relative to mean.
    """
    vals = np.abs(np.asarray(sensitivity).ravel())
    
    if len(vals) == 0:
        return np.inf
    
    mean_val = np.mean(vals)
    std_val = np.std(vals)
    min_val = np.min(vals)
    
    if mean_val < np.finfo(float).eps:
        return np.inf
    
    # Coefficient of variation (0 = perfectly uniform, >1 = high variation)
    coeff_of_variation = std_val / mean_val
    
    # Penalty for minimum values being too low relative to mean
    # We want min_val to be close to mean_val (ideally min_val/mean_val ≈ 0.8-1.0)
    min_ratio = min_val / mean_val
    min_penalty = max(0, 1.0 - min_ratio) ** 2  # Quadratic penalty
    
    # Combined score: favor uniformity AND ensure no extremely low values
    uniformity_error = coeff_of_variation + 2.0 * min_penalty
    
    return float(uniformity_error)


def evaluate_state(fwd, state, variable_mask, low_cond, high_cond):
    """Evaluate element state by computing sensitivity uniformity score."""
    perm = make_permittivity(
        state_high=state,
        variable_mask=variable_mask,
        low_cond=low_cond,
        high_cond=high_cond,
    )

    jacobian, _ = fwd.compute_jac(perm=perm)
    sensitivity = element_sensitivity_from_jacobian(jacobian)
    score = uniformity_score(sensitivity)

    return perm, sensitivity, score


def optimize_global_all_elements_ga(
    fwd,
    variable_mask,
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
):
    """Global optimization of element states using genetic algorithm with entropy objective."""
    rng = np.random.default_rng(seed)

    n_var = int(np.count_nonzero(variable_mask))
    generations = int(max(1, generations))
    pop_size = int(max(4, pop_size))
    elite_count = int(max(1, min(elite_count, pop_size - 1)))
    crossover_rate = float(min(max(crossover_rate, 0.0), 1.0))
    mutation_rate = float(min(max(mutation_rate, 0.0), 1.0))
    init_high_fraction = float(min(max(init_high_fraction, 0.0), 1.0))
    tournament_size = int(max(2, min(tournament_size, pop_size)))

    population = rng.random((pop_size, n_var)) < init_high_fraction

    best_state = None
    best_perm = None
    best_sensitivity = None
    best_score = np.inf

    best_history = []
    current_history = []

    for gen in range(1, generations + 1):
        evaluated = []
        for i in range(pop_size):
            state = population[i]
            perm, sensitivity, score = evaluate_state(
                fwd=fwd,
                state=state,
                variable_mask=variable_mask,
                low_cond=low_cond,
                high_cond=high_cond,
            )
            evaluated.append({
                "state": state.copy(),
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
            best_state = gen_best["state"].copy()
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
            next_population.append(evaluated[i]["state"].copy())

        # Tournament selection and reproduction
        def tournament_select():
            idxs = rng.choice(pop_size, size=tournament_size, replace=False)
            best_idx = min(idxs, key=lambda idx: evaluated[idx]["score"])
            return evaluated[best_idx]["state"].copy()

        while len(next_population) < pop_size:
            parent_a = tournament_select()
            parent_b = tournament_select()

            # Crossover
            if rng.random() < crossover_rate and n_var > 1:
                cx = int(rng.integers(1, n_var))
                child = np.concatenate([parent_a[:cx], parent_b[cx:]])
            else:
                child = parent_a.copy()

            # Mutation
            if mutation_rate > 0.0:
                mutation_mask = rng.random(n_var) < mutation_rate
                child[mutation_mask] = ~child[mutation_mask]

            next_population.append(child)

        population = np.asarray(next_population, dtype=bool)

    return {
        "best_state": best_state,
        "best_perm": best_perm,
        "best_sensitivity": best_sensitivity,
        "best_score": float(best_score),
        "best_history": np.asarray(best_history, dtype=float),
        "current_history": np.asarray(current_history, dtype=float),
        "n_variable": n_var,
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

    fig.suptitle("Global Binary Optimization (Uniformity Objective)", fontsize=13)

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

    # Run genetic algorithm optimization
    if args.pop_size < 10:
        print(
            "Warning: pop_size < 10 can converge prematurely; "
            "consider >= 20 for better global search."
        )

    result = optimize_global_all_elements_ga(
        fwd=fwd,
        variable_mask=variable_mask,
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
    )

    comparison_rows = []
    n_var = int(np.count_nonzero(variable_mask))
    tri = mesh_obj.element
    pts = mesh_obj.node
    centroids = np.mean(pts[tri], axis=1)
    variable_indices = np.where(variable_mask)[0]

    # Centered low-conductivity rectangle on high-conductivity background
    cx = 0.5 * (p1[0] + p2[0])
    cy = 0.5 * (p1[1] + p2[1])
    rect_w = 0.25 * (p2[0] - p1[0])
    rect_h = 0.50 * (p2[1] - p1[1])
    in_center_rect = (
        (np.abs(centroids[:, 0] - cx) <= rect_w / 2.0)
        & (np.abs(centroids[:, 1] - cy) <= rect_h / 2.0)
    )

    middle_low_state = np.ones(n_var, dtype=bool)
    middle_low_state[in_center_rect[variable_indices]] = False

    rect_perm, rect_sens, rect_score = evaluate_state(
        fwd=fwd,
        state=middle_low_state,
        variable_mask=variable_mask,
        low_cond=args.low_cond,
        high_cond=args.high_cond,
    )
    comparison_rows.append(
        {
            "name": "Center Low Rectangle",
            "perm": rect_perm,
            "sensitivity": rect_sens,
            "score": float(rect_score),
        }
    )

    all_low_state = np.zeros(n_var, dtype=bool)
    low_perm, low_sens, low_score = evaluate_state(
        fwd=fwd,
        state=all_low_state,
        variable_mask=variable_mask,
        low_cond=args.low_cond,
        high_cond=args.high_cond,
    )
    comparison_rows.append(
        {
            "name": "All Low Conductivity",
            "perm": low_perm,
            "sensitivity": low_sens,
            "score": float(low_score),
        }
    )

    best_sens = np.asarray(result["best_sensitivity"]).ravel()
    print(f"Minimum entropy found: {result['best_score']:.6e}")
    print(f"Best sensitivity min: {best_sens.min():.6e}")
    print(f"Best sensitivity max: {best_sens.max():.6e}")

    print(f"High elements in best state: {np.count_nonzero(result['best_state'])}")
    print(f"Low elements in best state: {result['n_variable'] - np.count_nonzero(result['best_state'])}")

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
