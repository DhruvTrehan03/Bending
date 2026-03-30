from __future__ import absolute_import, division, print_function

import argparse
import numpy as np
import matplotlib.pyplot as plt

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


def voltage_range(v):
    vals = np.asarray(v).ravel()
    return float(np.max(vals) - np.min(vals))


def element_sensitivity_from_jacobian(jacobian):
    return np.linalg.norm(np.asarray(jacobian), axis=0)


def sensitivity_uniformity_score(sensitivity, norm_scale=1.0):
    """Sensitivity uniformity objective (minimized by optimizer).

    Lower score means a more spatially uniform element sensitivity pattern.
    """
    vals = np.abs(np.asarray(sensitivity).ravel())
    eps = np.finfo(float).eps

    # Keep optional scaling for compatibility with CLI and previous runs.
    scale = max(float(norm_scale), eps)
    normalized = vals / scale

    mean_sens = float(np.mean(normalized))
    std_sens = float(np.std(normalized))

    # Coefficient of variation: dimensionless uniformity score.
    score = std_sens / max(mean_sens, eps)
    return float(score)


def sensitivity_objective_components(sensitivity, norm_scale=1.0):
    vals = np.abs(np.asarray(sensitivity).ravel())
    eps = np.finfo(float).eps

    scale = max(float(norm_scale), eps)
    normalized = vals / scale
    mean_sens = float(np.mean(normalized))
    std_sens = float(np.std(normalized))
    score = float(std_sens / max(mean_sens, eps))
    return mean_sens, std_sens, score


def is_sensitivity_uniformity_target(target):
    return target in {"sensitivity-uniformity", "sensitivity-matrix-uniformity"}


def evaluate_state(
    fwd,
    state,
    variable_mask,
    low_cond,
    high_cond,
    target,
    sens_norm_scale=1.0,
):
    perm = make_permittivity(
        state_high=state,
        variable_mask=variable_mask,
        low_cond=low_cond,
        high_cond=high_cond,
    )

    if target == "voltage-range":
        v = np.asarray(fwd.solve_eit(perm=perm)).ravel()
        sensitivity = None
        score = voltage_range(v)
    elif is_sensitivity_uniformity_target(target):
        jacobian, v0 = fwd.compute_jac(perm=perm)
        v = np.asarray(v0).ravel()
        sensitivity = element_sensitivity_from_jacobian(jacobian)
        score = sensitivity_uniformity_score(
            sensitivity,
            norm_scale=sens_norm_scale,
        )
    else:
        raise ValueError(f"Unknown target: {target}")

    return perm, v, sensitivity, score


def optimize_global_all_elements_ga(
    fwd,
    variable_mask,
    low_cond,
    high_cond,
    target,
    generations,
    pop_size,
    elite_count,
    crossover_rate,
    mutation_rate,
    init_high_fraction,
    tournament_size,
    seed,
    sens_norm_scale,
):
    """Global optimization over full element state using a genetic algorithm."""
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
    best_v = None
    best_sensitivity = None
    best_score = np.inf

    best_history = []
    current_history = []

    for gen in range(1, generations + 1):
        evaluated = []
        for i in range(pop_size):
            state = population[i]
            perm, v, sensitivity, score = evaluate_state(
                fwd=fwd,
                state=state,
                variable_mask=variable_mask,
                low_cond=low_cond,
                high_cond=high_cond,
                target=target,
                sens_norm_scale=sens_norm_scale,
            )
            evaluated.append({
                "state": state.copy(),
                "perm": perm,
                "v": np.asarray(v).copy(),
                "sensitivity": None if sensitivity is None else np.asarray(sensitivity).copy(),
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
            best_v = gen_best["v"].copy()
            best_sensitivity = None if gen_best["sensitivity"] is None else gen_best["sensitivity"].copy()

        best_history.append(float(best_score))
        current_history.append(float(gen_best_score))

        if gen % 10 == 0 or gen == generations:
            print(
                f"Gen {gen:4d}/{generations} | best_gen={gen_best_score:.6e} | "
                f"best_global={best_score:.6e} | mean={gen_mean_score:.6e}"
            )

        next_population = []

        # Elitism: carry top individuals unchanged.
        for i in range(elite_count):
            next_population.append(evaluated[i]["state"].copy())

        def tournament_select():
            idxs = rng.choice(pop_size, size=tournament_size, replace=False)
            best_idx = min(idxs, key=lambda idx: evaluated[idx]["score"])
            return evaluated[best_idx]["state"].copy()

        while len(next_population) < pop_size:
            parent_a = tournament_select()
            parent_b = tournament_select()

            if rng.random() < crossover_rate and n_var > 1:
                cx = int(rng.integers(1, n_var))
                child = np.concatenate([parent_a[:cx], parent_b[cx:]])
            else:
                child = parent_a.copy()

            if mutation_rate > 0.0:
                mutation_mask = rng.random(n_var) < mutation_rate
                child[mutation_mask] = ~child[mutation_mask]

            next_population.append(child)

        population = np.asarray(next_population, dtype=bool)

    return {
        "best_state": best_state,
        "best_perm": best_perm,
        "best_v": best_v,
        "best_sensitivity": best_sensitivity,
        "best_score": float(best_score),
        "best_history": np.asarray(best_history, dtype=float),
        "current_history": np.asarray(current_history, dtype=float),
        "n_variable": n_var,
    }


def optimize_global_all_elements(
    fwd,
    variable_mask,
    low_cond,
    high_cond,
    target,
    iterations,
    temperature,
    cooling,
    restarts,
    init_high_fraction,
    seed,
    sens_norm_scale,
):
    """Global optimization over the full element state using simulated annealing."""
    rng = np.random.default_rng(seed)

    n_var = int(np.count_nonzero(variable_mask))
    iterations = int(max(1, iterations))
    restarts = int(max(1, restarts))
    temperature = float(max(1e-9, temperature))
    cooling = float(min(max(cooling, 1e-6), 1.0))
    init_high_fraction = float(min(max(init_high_fraction, 0.0), 1.0))

    global_best_score = np.inf
    global_best = None
    best_history = []
    current_history = []

    for restart_idx in range(1, restarts + 1):
        state = rng.random(n_var) < init_high_fraction
        perm, v, sensitivity, score = evaluate_state(
            fwd=fwd,
            state=state,
            variable_mask=variable_mask,
            low_cond=low_cond,
            high_cond=high_cond,
            target=target,
            sens_norm_scale=sens_norm_scale,
        )

        restart_best_state = state.copy()
        restart_best_perm = perm.copy()
        restart_best_v = np.asarray(v).copy()
        restart_best_sensitivity = None if sensitivity is None else np.asarray(sensitivity).copy()
        restart_best_score = float(score)

        t = temperature
        for i in range(1, iterations + 1):
            cand_state = state.copy()
            n_flips = int(rng.integers(1, 4))
            flip_idx = rng.choice(n_var, size=min(n_flips, n_var), replace=False)
            cand_state[flip_idx] = ~cand_state[flip_idx]

            cand_perm, cand_v, cand_sensitivity, cand_score = evaluate_state(
                fwd=fwd,
                state=cand_state,
                variable_mask=variable_mask,
                low_cond=low_cond,
                high_cond=high_cond,
                target=target,
                sens_norm_scale=sens_norm_scale,
            )

            delta = cand_score - score
            accept = (delta <= 0.0) or (rng.random() < np.exp(-delta / max(t, 1e-12)))
            if accept:
                state = cand_state
                perm = cand_perm
                v = cand_v
                sensitivity = cand_sensitivity
                score = cand_score

                if score < restart_best_score:
                    restart_best_state = state.copy()
                    restart_best_perm = perm.copy()
                    restart_best_v = np.asarray(v).copy()
                    restart_best_sensitivity = None if sensitivity is None else np.asarray(sensitivity).copy()
                    restart_best_score = float(score)

            current_history.append(float(score))
            best_history.append(float(min(global_best_score, restart_best_score)))

            if i % 50 == 0 or i == iterations:
                print(
                    f"Restart {restart_idx:2d}/{restarts} | iter {i:5d}/{iterations} | "
                    f"current={score:.6e} | best_restart={restart_best_score:.6e}"
                )

            t *= cooling

        if restart_best_score < global_best_score:
            global_best_score = restart_best_score
            global_best = {
                "best_state": restart_best_state,
                "best_perm": restart_best_perm,
                "best_v": restart_best_v,
                "best_sensitivity": restart_best_sensitivity,
                "best_score": global_best_score,
                "n_variable": n_var,
            }
            print(f"  -> New global best after restart {restart_idx}: {global_best_score:.6e}")

    if global_best is None:
        state = np.zeros(n_var, dtype=bool)
        perm, v, sensitivity, score = evaluate_state(
            fwd=fwd,
            state=state,
            variable_mask=variable_mask,
            low_cond=low_cond,
            high_cond=high_cond,
            target=target,
            sens_norm_scale=sens_norm_scale,
        )
        global_best = {
            "best_state": state,
            "best_perm": perm,
            "best_v": np.asarray(v),
            "best_sensitivity": sensitivity,
            "best_score": float(score),
            "n_variable": n_var,
        }
        current_history = [float(score)]
        best_history = [float(score)]

    global_best["best_history"] = np.asarray(best_history, dtype=float)
    global_best["current_history"] = np.asarray(current_history, dtype=float)
    return global_best


def plot_results(
    mesh_obj,
    p1,
    p2,
    best_perm,
    best_v,
    best_sensitivity,
    best_hist,
    curr_hist,
    target,
    best_score,
    comparison_rows=None,
):
    """Show best conductivity map, objective output, and optimization progress."""
    pts = mesh_obj.node
    tri = mesh_obj.element
    el_pos = mesh_obj.el_pos

    xs, ys = pts[:, 0], pts[:, 1]

    comparison_rows = comparison_rows or []
    rows = [
        {
            "name": "Best Optimized",
            "perm": best_perm,
            "v": best_v,
            "sensitivity": best_sensitivity,
            "score": best_score,
        }
    ] + comparison_rows

    fig, axes = plt.subplots(len(rows), 3, figsize=(18, 5.2 * len(rows)))
    if len(rows) == 1:
        axes = np.asarray([axes])

    if target == "voltage-range":
        fig.suptitle("Global Binary Optimization (Voltage Range Objective)", fontsize=13)
    else:
        fig.suptitle("Global Binary Optimization (Sensitivity Objective)", fontsize=13)

    for r, row in enumerate(rows):
        ax0 = axes[r, 0]
        ax1 = axes[r, 1]
        ax2 = axes[r, 2]

        perm_row = np.asarray(row["perm"]).ravel()
        v_row = np.asarray(row["v"]).ravel()
        sens_row = None if row["sensitivity"] is None else np.asarray(row["sensitivity"]).ravel()

        im0 = ax0.tripcolor(xs, ys, tri, perm_row, shading="flat", cmap="RdBu_r")
        ax0.plot(xs[el_pos], ys[el_pos], "ko", markersize=3)
        ax0.set_aspect("equal")
        ax0.set_xlim(p1[0] - 0.2, p2[0] + 0.2)
        ax0.set_ylim(p1[1] - 0.2, p2[1] + 0.2)
        ax0.set_title(f"{row['name']} Conductivity")
        ax0.set_xlabel("x")
        ax0.set_ylabel("y")
        fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04, label="Conductivity")

        if is_sensitivity_uniformity_target(target) and sens_row is not None:
            im1 = ax1.tripcolor(xs, ys, tri, sens_row, shading="flat", cmap="viridis")
            ax1.set_aspect("equal")
            ax1.set_xlim(p1[0] - 0.2, p2[0] + 0.2)
            ax1.set_ylim(p1[1] - 0.2, p2[1] + 0.2)
            ax1.set_title(f"{row['name']} Sensitivity")
            ax1.set_xlabel("x")
            ax1.set_ylabel("y")
            fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label="||J[:, elem]||2")
        else:
            ax1.plot(np.arange(v_row.size), v_row, linewidth=1.6)
            ax1.set_title(f"{row['name']} Voltage Output")
            ax1.set_xlabel("Measurement index")
            ax1.set_ylabel("Voltage")
            ax1.grid(True, alpha=0.3)

        if r == 0:
            ax2.plot(best_hist, label="Best-so-far score", linewidth=1.8)
            ax2.plot(curr_hist, label="Current score", linewidth=1.0, alpha=0.7)
            ax2.set_title("Optimization Progress")
            ax2.set_xlabel("Iteration")
            ax2.set_ylabel("Objective score")
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc="best")
        else:
            ax2.axis("off")
            ax2.text(
                0.03,
                0.85,
                f"{row['name']}\nObjective score: {row['score']:.6e}\n"
                f"Sensitivity min: {np.min(sens_row):.6e}\n"
                f"Sensitivity max: {np.max(sens_row):.6e}",
                va="top",
                ha="left",
                fontsize=10,
            )

    plt.tight_layout()
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Global optimization over all rectangle elements (binary high/low conductivity)."
    )
    parser.add_argument(
        "--goal",
        type=str,
        required=True,
        choices=["voltage-range", "sensitivity-uniformity", "sensitivity-matrix-uniformity"],
        help="Optimization goal/objective.",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="anneal",
        choices=["anneal", "ga"],
        help="Global optimization algorithm: simulated annealing or genetic algorithm.",
    )
    parser.add_argument("--iters", type=int, default=1200, help="Anneal iterations or GA generations")
    parser.add_argument("--restarts", type=int, default=3, help="Number of random restarts")
    parser.add_argument("--temp", type=float, default=0.05, help="Initial temperature")
    parser.add_argument("--cooling", type=float, default=0.995, help="Cooling multiplier in (0, 1]")
    parser.add_argument("--init-high-fraction", type=float, default=0.5, help="Initial high fraction")
    parser.add_argument("--ga-pop-size", type=int, default=24, help="Population size for GA")
    parser.add_argument("--ga-elite", type=int, default=2, help="Elite count kept each GA generation")
    parser.add_argument("--ga-crossover", type=float, default=0.8, help="Crossover probability for GA")
    parser.add_argument("--ga-mutation", type=float, default=0.01, help="Per-gene mutation probability for GA")
    parser.add_argument("--ga-tournament", type=int, default=3, help="Tournament size for GA selection")
    parser.add_argument(
        "--ga-runs",
        type=int,
        default=1,
        help="Number of independent GA runs (different seeds); best run is kept",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--low", type=float, default=1e-6, help="Low conductivity value")
    parser.add_argument("--high", type=float, default=1e6, help="High conductivity value")
    parser.add_argument("--h0", type=float, default=0.1, help="Mesh element size")
    parser.add_argument(
        "--sens-norm-scale",
        type=float,
        default=None,
        help="Fixed normalization scale for sensitivity objective; if omitted, auto-estimated from baseline",
    )
    return parser.parse_args()


def run():
    args = parse_args()

    mesh_obj, protocol_obj, variable_mask, p1, p2 = build_model(h0=args.h0)
    fwd = EITForward(mesh_obj, protocol_obj)

    print(f"Total elements: {mesh_obj.element.shape[0]}")
    print(f"Optimized variable elements: {np.count_nonzero(variable_mask)}")

    sens_norm_scale = 1.0
    if is_sensitivity_uniformity_target(args.goal):
        if args.sens_norm_scale is not None:
            sens_norm_scale = max(float(args.sens_norm_scale), np.finfo(float).eps)
        else:
            # Baseline robust scale from all-low conductivity map.
            base_state = np.zeros(int(np.count_nonzero(variable_mask)), dtype=bool)
            _, _, base_sens, _ = evaluate_state(
                fwd=fwd,
                state=base_state,
                variable_mask=variable_mask,
                low_cond=args.low,
                high_cond=args.high,
                target=args.goal,
                sens_norm_scale=1.0,
            )
            sens_norm_scale = max(float(np.percentile(np.abs(base_sens), 90.0)), np.finfo(float).eps)
        print(f"Sensitivity normalization scale: {sens_norm_scale:.6e}")

    if args.algorithm == "anneal":
        result = optimize_global_all_elements(
            fwd=fwd,
            variable_mask=variable_mask,
            low_cond=args.low,
            high_cond=args.high,
            target=args.goal,
            iterations=args.iters,
            temperature=args.temp,
            cooling=args.cooling,
            restarts=args.restarts,
            init_high_fraction=args.init_high_fraction,
            seed=args.seed,
            sens_norm_scale=sens_norm_scale,
        )
    else:
        ga_runs = int(max(1, args.ga_runs))
        if args.ga_pop_size < 10:
            print(
                "Warning: ga-pop-size < 10 can converge prematurely; "
                "consider >= 20 for better global search."
            )

        best_result = None
        for run_idx in range(ga_runs):
            run_seed = int(args.seed + run_idx)
            print(f"GA run {run_idx + 1}/{ga_runs} with seed={run_seed}")

            run_result = optimize_global_all_elements_ga(
                fwd=fwd,
                variable_mask=variable_mask,
                low_cond=args.low,
                high_cond=args.high,
                target=args.goal,
                generations=args.iters,
                pop_size=args.ga_pop_size,
                elite_count=args.ga_elite,
                crossover_rate=args.ga_crossover,
                mutation_rate=args.ga_mutation,
                init_high_fraction=args.init_high_fraction,
                tournament_size=args.ga_tournament,
                seed=run_seed,
                sens_norm_scale=sens_norm_scale,
            )

            if best_result is None or run_result["best_score"] < best_result["best_score"]:
                best_result = run_result
                print(
                    f"  -> New best GA run: score={best_result['best_score']:.6e} "
                    f"(seed={run_seed})"
                )

        result = best_result

    comparison_rows = []
    if is_sensitivity_uniformity_target(args.goal):
        n_var = int(np.count_nonzero(variable_mask))
        tri = mesh_obj.element
        pts = mesh_obj.node
        centroids = np.mean(pts[tri], axis=1)
        variable_indices = np.where(variable_mask)[0]

        # Build a centered low-conductivity rectangle on a high-conductivity background.
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

        rect_perm, rect_v, rect_sens, rect_score = evaluate_state(
            fwd=fwd,
            state=middle_low_state,
            variable_mask=variable_mask,
            low_cond=args.low,
            high_cond=args.high,
            target=args.goal,
            sens_norm_scale=sens_norm_scale,
        )
        comparison_rows.append(
            {
                "name": "Center Low Rectangle",
                "perm": rect_perm,
                "v": rect_v,
                "sensitivity": rect_sens,
                "score": float(rect_score),
            }
        )

        all_low_state = np.zeros(n_var, dtype=bool)
        low_perm, low_v, low_sens, low_score = evaluate_state(
            fwd=fwd,
            state=all_low_state,
            variable_mask=variable_mask,
            low_cond=args.low,
            high_cond=args.high,
            target=args.goal,
            sens_norm_scale=sens_norm_scale,
        )
        comparison_rows.append(
            {
                "name": "All Low Conductivity",
                "perm": low_perm,
                "v": low_v,
                "sensitivity": low_sens,
                "score": float(low_score),
            }
        )

    best_v = np.asarray(result["best_v"]).ravel()
    if args.goal == "voltage-range":
        print(f"Minimum voltage range found: {result['best_score']:.6e}")
        print(f"Best absolute voltage min: {best_v.min():.6e}")
        print(f"Best absolute voltage max: {best_v.max():.6e}")
    else:
        best_sens = np.asarray(result["best_sensitivity"]).ravel()
        mean_sens, std_sens, recomputed_score = sensitivity_objective_components(best_sens, sens_norm_scale)
        print(f"Minimum sensitivity objective score found: {result['best_score']:.6e}")
        print(f"Normalized sensitivity mean: {mean_sens:.6e}")
        print(f"Normalized sensitivity std: {std_sens:.6e}")
        print(f"Best sensitivity min: {best_sens.min():.6e}")
        print(f"Best sensitivity max: {best_sens.max():.6e}")
        print(
            f"Best sensitivity objective recomputed (uniformity CV): "
            f"{recomputed_score:.6e}"
        )

    print(f"High elements in best state: {np.count_nonzero(result['best_state'])}")
    print(f"Low elements in best state: {result['n_variable'] - np.count_nonzero(result['best_state'])}")

    plot_results(
        mesh_obj=mesh_obj,
        p1=p1,
        p2=p2,
        best_perm=result["best_perm"],
        best_v=result["best_v"],
        best_sensitivity=result["best_sensitivity"],
        best_hist=result["best_history"],
        curr_hist=result["current_history"],
        target=args.goal,
        best_score=result["best_score"],
        comparison_rows=comparison_rows,
    )


def main():
    run()


if __name__ == "__main__":
    main()
