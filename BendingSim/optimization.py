"""Standalone optimization script built on mesh_generation and benchmarking."""

from __future__ import annotations

import argparse
import configparser
import csv
import json
import os
from datetime import datetime

import numpy as np
from scipy.optimize import differential_evolution

from pyeit.eit.fem import EITForward

try:
    from mesh_generation import (
        build_element_adjacency,
        build_electrode_support_elements,
        build_field_geometry,
        build_model,
        parameter_vector_size_from_cfg,
        parameterized_state_from_theta,
    )
    from benchmarking import (
        count_isolated_high_elements,
        electrodes_connected_by_high_region,
        evaluate_state,
        expand_state_to_full_mesh,
        get_profile_weights,
        get_connected_components,
        get_state_for_connectivity,
        has_connected_high_elements,
        list_benchmarks,
        plot_results,
        validate_high_element_connectivity,
    )
except ImportError:
    from .mesh_generation import (
        build_element_adjacency,
        build_electrode_support_elements,
        build_field_geometry,
        build_model,
        parameter_vector_size_from_cfg,
        parameterized_state_from_theta,
    )
    from .benchmarking import (
        count_isolated_high_elements,
        electrodes_connected_by_high_region,
        evaluate_state,
        expand_state_to_full_mesh,
        get_profile_weights,
        get_connected_components,
        get_state_for_connectivity,
        has_connected_high_elements,
        list_benchmarks,
        plot_results,
        validate_high_element_connectivity,
    )


def _clean_cfg_string(value, default=""):
    if value is None:
        return str(default).strip().lower()
    text = str(value)
    for sep in ("#", ";"):
        if sep in text:
            text = text.split(sep, 1)[0]
    return text.strip().lower()


def _build_run_name(args):
    custom = str(getattr(args, "run_name", "") or "").strip()
    if custom:
        return custom

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{args.optimizer}_{args.benchmark_name}_{stamp}"


def _save_run_artifacts(
    output_dir,
    run_name,
    mesh_obj,
    p1,
    p2,
    result,
    args,
    field_shape_cfg,
    benchmark_weights,
    connectivity_info,
):
    os.makedirs(output_dir, exist_ok=True)

    base_path = os.path.join(output_dir, run_name)
    npz_path = f"{base_path}.npz"
    json_path = f"{base_path}.json"
    history_csv_path = f"{base_path}_history.csv"
    elements_csv_path = f"{base_path}_elements.csv"

    best_theta = result["best_theta"]
    if best_theta is None:
        best_theta = np.empty((0,), dtype=float)

    np.savez_compressed(
        npz_path,
        node=np.asarray(mesh_obj.node, dtype=float),
        element=np.asarray(mesh_obj.element, dtype=int),
        el_pos=np.asarray(mesh_obj.el_pos, dtype=int),
        p1=np.asarray(p1, dtype=float),
        p2=np.asarray(p2, dtype=float),
        best_perm=np.asarray(result["best_perm"], dtype=float),
        best_sensitivity=np.asarray(result["best_sensitivity"], dtype=float),
        best_state=np.asarray(result["best_state"], dtype=bool),
        best_theta=np.asarray(best_theta, dtype=float),
        best_history=np.asarray(result["best_history"], dtype=float),
        current_history=np.asarray(result["current_history"], dtype=float),
    )

    with open(history_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "best_score", "current_score"])
        best_history = np.asarray(result["best_history"], dtype=float).ravel()
        current_history = np.asarray(result["current_history"], dtype=float).ravel()
        for idx, (best_val, curr_val) in enumerate(zip(best_history, current_history), start=1):
            writer.writerow([idx, float(best_val), float(curr_val)])

    with open(elements_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["element_index", "conductivity", "sensitivity", "is_high"])
        perm = np.asarray(result["best_perm"], dtype=float).ravel()
        sens = np.asarray(result["best_sensitivity"], dtype=float).ravel()
        high = np.asarray(result["best_state"], dtype=bool).ravel()
        for idx in range(perm.size):
            writer.writerow([idx, float(perm[idx]), float(sens[idx]), int(high[idx])])

    metadata = {
        "run_name": run_name,
        "optimizer": str(args.optimizer),
        "benchmark_name": str(args.benchmark_name),
        "benchmark_profile": str(args.benchmark_profile),
        "benchmark_weights": {
            "entropy": float(benchmark_weights.get("entropy", 0.0)),
            "isolated_penalty": float(benchmark_weights.get("isolated_penalty", 0.0)),
            "disconnected_penalty": float(benchmark_weights.get("disconnected_penalty", 0.0)),
        },
        "seed": int(args.seed),
        "h0": float(args.h0),
        "low_cond": float(args.low_cond),
        "high_cond": float(args.high_cond),
        "n_elements": int(np.asarray(mesh_obj.element).shape[0]),
        "n_electrodes": int(np.asarray(mesh_obj.el_pos).size),
        "best_score": float(result["best_score"]),
        "best_sensitivity_min": float(np.min(np.asarray(result["best_sensitivity"], dtype=float))),
        "best_sensitivity_max": float(np.max(np.asarray(result["best_sensitivity"], dtype=float))),
        "p1": [float(v) for v in p1],
        "p2": [float(v) for v in p2],
        "connectivity": {
            "high_elements_connected": bool(connectivity_info["is_connected"]),
            "electrodes_connected_by_high_region": bool(connectivity_info["electrodes_linked"]),
            "high_connected_components": int(connectivity_info["components"]),
            "isolated_high_elements": int(connectivity_info["isolated_count"]),
        },
        "field_shape_cfg": {
            key: (
                [float(x) for x in np.asarray(val).ravel()]
                if isinstance(val, (list, tuple, np.ndarray))
                else bool(val)
                if isinstance(val, (bool, np.bool_))
                else int(val)
                if isinstance(val, (int, np.integer))
                else float(val)
                if isinstance(val, (float, np.floating))
                else str(val)
            )
            for key, val in field_shape_cfg.items()
            if key not in {"branching_electrode_points_norm", "branching_electrode_points_norm_right"}
        },
        "artifact_files": {
            "npz": os.path.basename(npz_path),
            "metadata_json": os.path.basename(json_path),
            "history_csv": os.path.basename(history_csv_path),
            "elements_csv": os.path.basename(elements_csv_path),
        },
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return {
        "npz": npz_path,
        "metadata_json": json_path,
        "history_csv": history_csv_path,
        "elements_csv": elements_csv_path,
    }


def save_run_artifacts(
    output_dir,
    run_name,
    mesh_obj,
    p1,
    p2,
    result,
    args,
    field_shape_cfg,
    benchmark_weights,
    connectivity_info,
):
    """Public wrapper for saving optimization artifacts."""
    return _save_run_artifacts(
        output_dir=output_dir,
        run_name=run_name,
        mesh_obj=mesh_obj,
        p1=p1,
        p2=p2,
        result=result,
        args=args,
        field_shape_cfg=field_shape_cfg,
        benchmark_weights=benchmark_weights,
        connectivity_info=connectivity_info,
    )


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
    tournament_size,
    seed,
    adjacency=None,
    electrode_supports=None,
    enforce_connectivity=True,
    repair_disconnected=False,
    benchmark_fn=None,
    benchmark_name=None,
    benchmark_weights=None,
    progress_callback=None,
):
    rng = np.random.default_rng(seed)
    mesh_centroids = None
    if isinstance(field_geometry, dict):
        mesh_centroids = field_geometry.get("centroids")

    model = str(field_shape_cfg.get("model", "rbf")).strip().lower()
    
    # Determine number of parameters based on model type
    if model == "element":
        # For element-wise optimization: one param per variable element
        n_params = int(np.count_nonzero(variable_mask))
    else:
        n_params = parameter_vector_size_from_cfg(field_shape_cfg, field_geometry)
    
    if n_params == 0:
        raise ValueError("No parameters available for optimization. Check mesh and variable_mask.")
    
    generations = int(max(1, generations))
    pop_size = int(max(4, pop_size))
    elite_count = int(max(1, min(elite_count, pop_size - 1)))
    crossover_rate = float(min(max(crossover_rate, 0.0), 1.0))
    mutation_rate = float(min(max(mutation_rate, 0.0), 1.0))
    tournament_size = int(max(2, min(tournament_size, pop_size)))

    population = rng.normal(loc=0.0, scale=0.25, size=(pop_size, n_params))
    if model == "rbf":
        population[:, 0] = np.clip(
            rng.normal(loc=float(field_shape_cfg["rbf_bias_init"]), scale=0.20, size=pop_size),
            -1.0,
            1.0,
        )
        if n_params > 1:
            population[:, 1:] = np.clip(
                rng.normal(
                    loc=0.0,
                    scale=float(field_shape_cfg["rbf_weight_scale"]),
                    size=(pop_size, n_params - 1),
                ),
                -1.0,
                1.0,
            )
    else:
        population = np.clip(rng.normal(loc=0.0, scale=0.40, size=(pop_size, n_params)), -1.0, 1.0)
        population[:, 0] = np.clip(
            rng.normal(loc=float(field_shape_cfg["fractal_threshold_init"]), scale=0.25, size=pop_size),
            -1.0,
            1.0,
        )

    best_state = None
    best_theta = None
    best_field = None
    best_perm = None
    best_sensitivity = None
    best_score = np.inf

    best_history = []
    current_history = []

    for gen in range(1, generations + 1):
        evaluated = []
        for i in range(pop_size):
            theta = population[i]
            state, field = parameterized_state_from_theta(theta, field_geometry, field_shape_cfg)
            perm, sensitivity, score = evaluate_state(
                fwd=fwd,
                state=state,
                variable_mask=variable_mask,
                low_cond=low_cond,
                high_cond=high_cond,
                adjacency=adjacency,
                electrode_supports=electrode_supports,
                enforce_connectivity=enforce_connectivity,
                repair_disconnected=repair_disconnected,
                benchmark_fn=benchmark_fn,
                benchmark_name=benchmark_name,
                benchmark_weights=benchmark_weights,
                mesh_centroids=mesh_centroids,
            )
            evaluated.append(
                {
                    "theta": theta.copy(),
                    "state": state.copy(),
                    "field": np.asarray(field).copy(),
                    "perm": np.asarray(perm).copy(),
                    "sensitivity": np.asarray(sensitivity).copy(),
                    "score": float(score),
                }
            )

        evaluated.sort(key=lambda d: d["score"])
        gen_best = evaluated[0]
        gen_best_score = gen_best["score"]

        if best_state is None or gen_best_score <= best_score:
            best_score = gen_best_score
            best_theta = gen_best["theta"].copy()
            best_state = gen_best["state"].copy()
            best_field = gen_best["field"].copy()
            best_perm = gen_best["perm"].copy()
            best_sensitivity = gen_best["sensitivity"].copy()

        best_history.append(float(best_score))
        current_history.append(float(gen_best_score))

        if progress_callback is not None:
            progress_callback(
                {
                    "generation": gen,
                    "n_generations": generations,
                    "best_score": float(best_score),
                    "current_score": float(gen_best_score),
                    "best_theta": None if best_theta is None else np.asarray(best_theta, dtype=float).copy(),
                    "best_state": None if best_state is None else np.asarray(best_state, dtype=bool).copy(),
                    "best_field": None if best_field is None else np.asarray(best_field, dtype=float).copy(),
                    "best_perm": None if best_perm is None else np.asarray(best_perm, dtype=float).copy(),
                    "best_sensitivity": None if best_sensitivity is None else np.asarray(best_sensitivity, dtype=float).copy(),
                    "current_theta": np.asarray(gen_best["theta"], dtype=float).copy(),
                    "current_state": np.asarray(gen_best["state"], dtype=bool).copy(),
                    "current_field": np.asarray(gen_best["field"], dtype=float).copy(),
                    "current_perm": np.asarray(gen_best["perm"], dtype=float).copy(),
                    "current_sensitivity": np.asarray(gen_best["sensitivity"], dtype=float).copy(),
                    "n_params": int(n_params),
                    "n_variable": int(np.count_nonzero(variable_mask)),
                }
            )

        if gen % 10 == 0 or gen == generations:
            print(f"Gen {gen:4d}/{generations} | best_gen={gen_best_score:.6e} | best_global={best_score:.6e}")

        next_population = []
        for i in range(elite_count):
            next_population.append(evaluated[i]["theta"].copy())

        def tournament_select():
            idxs = rng.choice(pop_size, size=tournament_size, replace=False)
            winner = min(idxs, key=lambda idx: evaluated[idx]["score"])
            return evaluated[winner]["theta"].copy()

        while len(next_population) < pop_size:
            parent_a = tournament_select()
            parent_b = tournament_select()

            if rng.random() < crossover_rate and n_params > 1:
                blend = rng.random(n_params)
                child = blend * parent_a + (1.0 - blend) * parent_b
            else:
                child = parent_a.copy()

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
        "best_field": best_field,
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
    enforce_connectivity=True,
    repair_disconnected=False,
    benchmark_fn=None,
    benchmark_name=None,
    benchmark_weights=None,
    progress_callback=None,
):
    # Determine number of parameters based on model type
    model = str(field_shape_cfg.get("model", "rbf")).strip().lower()
    mesh_centroids = None
    if isinstance(field_geometry, dict):
        mesh_centroids = field_geometry.get("centroids")
    if model == "element":
        # For element-wise optimization: one param per variable element
        n_params = int(np.count_nonzero(variable_mask))
    else:
        n_params = parameter_vector_size_from_cfg(field_shape_cfg, field_geometry)
    
    if n_params == 0:
        raise ValueError("No parameters available for optimization. Check mesh and variable_mask.")
    
    maxiter = int(max(1, maxiter))
    popsize = int(max(2, popsize))
    recombination = float(np.clip(recombination, 0.0, 1.0))

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

    def evaluate_theta(theta):
        nonlocal best_state, best_theta, best_field, best_perm, best_sensitivity, best_score

        state, field = parameterized_state_from_theta(theta, field_geometry, field_shape_cfg)
        perm, sensitivity, score = evaluate_state(
            fwd=fwd,
            state=state,
            variable_mask=variable_mask,
            low_cond=low_cond,
            high_cond=high_cond,
            adjacency=adjacency,
            electrode_supports=electrode_supports,
            enforce_connectivity=enforce_connectivity,
            repair_disconnected=repair_disconnected,
            benchmark_fn=benchmark_fn,
            benchmark_name=benchmark_name,
            benchmark_weights=benchmark_weights,
            mesh_centroids=mesh_centroids,
        )

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
        current_score = evaluate_theta(xk)
        current_history.append(float(current_score))
        best_history.append(float(best_score))

        if progress_callback is not None:
            progress_callback(
                {
                    "generation": generation_counter["value"],
                    "n_generations": maxiter,
                    "best_score": float(best_score),
                    "current_score": float(current_score),
                    "best_theta": None if best_theta is None else np.asarray(best_theta, dtype=float).copy(),
                    "best_state": None if best_state is None else np.asarray(best_state, dtype=bool).copy(),
                    "best_field": None if best_field is None else np.asarray(best_field, dtype=float).copy(),
                    "best_perm": None if best_perm is None else np.asarray(best_perm, dtype=float).copy(),
                    "best_sensitivity": None if best_sensitivity is None else np.asarray(best_sensitivity, dtype=float).copy(),
                    "current_theta": np.asarray(xk, dtype=float).copy(),
                    "n_params": int(n_params),
                    "n_variable": int(np.count_nonzero(variable_mask)),
                }
            )

        gen = generation_counter["value"]
        if gen % 10 == 0 or gen == maxiter:
            print(f"DE gen {gen:4d}/{maxiter} | best_global={best_score:.6e} | current={current_score:.6e}")

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


def load_config(config_file="config.ini"):
    config = configparser.ConfigParser()

    if not os.path.exists(config_file):
        print(f"Config file {config_file} not found. Creating with defaults...")
        create_default_config(config_file)

    config.read(config_file)
    cfg = type("Config", (), {})()

    def _has(section, option):
        return config.has_option(section, option)

    def _get(section, option, fallback=None):
        if _has(section, option):
            return config.get(section, option)
        return fallback

    def _getint(primary_section, option, fallback, combined_option=None, legacy_sections=()):
        if _has(primary_section, option):
            return config.getint(primary_section, option)
        comb_key = combined_option or option
        if _has("combined", comb_key):
            return config.getint("combined", comb_key)
        for legacy in legacy_sections:
            if _has(legacy, option):
                return config.getint(legacy, option)
        return fallback

    def _getfloat(primary_section, option, fallback, combined_option=None, legacy_sections=()):
        if _has(primary_section, option):
            return config.getfloat(primary_section, option)
        comb_key = combined_option or option
        if _has("combined", comb_key):
            return config.getfloat("combined", comb_key)
        for legacy in legacy_sections:
            if _has(legacy, option):
                return config.getfloat(legacy, option)
        return fallback

    def _getbool(primary_section, option, fallback, combined_option=None, legacy_sections=()):
        if _has(primary_section, option):
            return config.getboolean(primary_section, option)
        comb_key = combined_option or option
        if _has("combined", comb_key):
            return config.getboolean("combined", comb_key)
        for legacy in legacy_sections:
            if _has(legacy, option):
                return config.getboolean(legacy, option)
        return fallback

    def _getclean(primary_section, option, fallback, combined_option=None, legacy_sections=()):
        val = _get(primary_section, option, fallback=None)
        if val is None:
            comb_key = combined_option or option
            val = _get("combined", comb_key, fallback=None)
        if val is None:
            for legacy in legacy_sections:
                val = _get(legacy, option, fallback=None)
                if val is not None:
                    break
        if val is None:
            val = fallback
        return _clean_cfg_string(val, default=fallback)

    cfg.generations = _getint("optimization", "generations", 100)
    cfg.pop_size = _getint("optimization", "pop_size", 24)
    cfg.elite_count = _getint("optimization", "elite_count", 2)
    cfg.crossover_rate = _getfloat("optimization", "crossover_rate", 0.8)
    cfg.mutation_rate = _getfloat("optimization", "mutation_rate", 0.01)
    cfg.tournament_size = _getint("optimization", "tournament_size", 3)
    cfg.optimizer = _getclean("optimization", "optimizer", "ga")
    cfg.de_maxiter = _getint("optimization", "de_maxiter", 25)
    cfg.de_popsize = _getint("optimization", "de_popsize", 6)
    cfg.de_mutation_min = _getfloat("optimization", "de_mutation_min", 0.5)
    cfg.de_mutation_max = _getfloat("optimization", "de_mutation_max", 1.0)
    cfg.de_recombination = _getfloat("optimization", "de_recombination", 0.7)

    cfg.seed = _getint(
        "combined",
        "seed",
        7,
        legacy_sections=("initial_conditions",),
    )

    cfg.low_cond = _getfloat(
        "combined",
        "low_cond",
        100,
        legacy_sections=("conductivity",),
    )
    cfg.high_cond = _getfloat(
        "combined",
        "high_cond",
        10000,
        legacy_sections=("conductivity",),
    )

    cfg.h0 = _getfloat(
        "combined",
        "h0",
        0.1,
        legacy_sections=("mesh",),
    )

    cfg.field_model = _getclean(
        "optimization",
        "field_model",
        "rbf",
        combined_option="field_model",
        legacy_sections=("field_shape",),
    )
    cfg.rbf_rows = _getint("optimization", "rbf_rows", 4, legacy_sections=("field_shape",))
    cfg.rbf_cols = _getint("optimization", "rbf_cols", 5, legacy_sections=("field_shape",))
    cfg.rbf_sigma_frac = _getfloat("optimization", "rbf_sigma_frac", 0.90, legacy_sections=("field_shape",))
    cfg.rbf_bias_init = _getfloat("optimization", "rbf_bias_init", 0.25, legacy_sections=("field_shape",))
    cfg.rbf_weight_scale = _getfloat("optimization", "rbf_weight_scale", 0.35, legacy_sections=("field_shape",))
    cfg.fractal_iter_max = _getint("optimization", "fractal_iter_max", 40, legacy_sections=("field_shape",))
    cfg.fractal_type = _getclean("optimization", "fractal_type", "branching", legacy_sections=("field_shape",))
    cfg.fractal_power = _getint("optimization", "fractal_power", 3, legacy_sections=("field_shape",))
    cfg.fractal_shift_frac = _getfloat("optimization", "fractal_shift_frac", 0.55, legacy_sections=("field_shape",))
    cfg.fractal_threshold_init = _getfloat("optimization", "fractal_threshold_init", 0.0, legacy_sections=("field_shape",))
    cfg.fractal_center_origin = _getbool("optimization", "fractal_center_origin", True, legacy_sections=("field_shape",))
    cfg.branching_depth_max = _getint("optimization", "branching_depth_max", 6, legacy_sections=("field_shape",))
    cfg.branching_max_children = _getint("optimization", "branching_max_children", 2, legacy_sections=("field_shape",))
    cfg.branching_angle_frac = _getfloat("optimization", "branching_angle_frac", 0.34, legacy_sections=("field_shape",))
    cfg.branching_length_decay = _getfloat("optimization", "branching_length_decay", 0.68, legacy_sections=("field_shape",))
    cfg.branching_width_frac = _getfloat("optimization", "branching_width_frac", 0.12, legacy_sections=("field_shape",))
    cfg.branching_width_decay = _getfloat("optimization", "branching_width_decay", 0.80, legacy_sections=("field_shape",))
    cfg.branching_mirror_vertical = _getbool("optimization", "branching_mirror_vertical", False, legacy_sections=("field_shape",))
    cfg.branching_aim_electrodes = _getbool("optimization", "branching_aim_electrodes", False, legacy_sections=("field_shape",))
    cfg.branching_target_blend = _getfloat("optimization", "branching_target_blend", 0.00, legacy_sections=("field_shape",))
    cfg.branching_root_x_frac = _getfloat("optimization", "branching_root_x_frac", 0.18, legacy_sections=("field_shape",))
    cfg.branching_root_y_frac = _getfloat("optimization", "branching_root_y_frac", 0.0, legacy_sections=("field_shape",))
    cfg.branching_seed_all_electrodes = _getbool("optimization", "branching_seed_all_electrodes", True, legacy_sections=("field_shape",))
    cfg.branching_force_meet_center = _getbool("optimization", "branching_force_meet_center", True, legacy_sections=("field_shape",))
    cfg.branching_meet_x_frac = _getfloat("optimization", "branching_meet_x_frac", 0.0, legacy_sections=("field_shape",))
    cfg.branching_meet_y_frac = _getfloat("optimization", "branching_meet_y_frac", 0.0, legacy_sections=("field_shape",))
    cfg.branching_meet_blend = _getfloat("optimization", "branching_meet_blend", 1.0, legacy_sections=("field_shape",))
    cfg.branching_random_angle_frac = _getfloat("optimization", "branching_random_angle_frac", 0.00, legacy_sections=("field_shape",))
    cfg.branching_random_center_boost = _getfloat("optimization", "branching_random_center_boost", 0.00, legacy_sections=("field_shape",))
    cfg.branching_random_center_power = _getfloat("optimization", "branching_random_center_power", 2.0, legacy_sections=("field_shape",))
    cfg.branching_force_touch_all_electrodes = _getbool("optimization", "branching_force_touch_all_electrodes", True, legacy_sections=("field_shape",))
    cfg.electrode_support_k = _getint("optimization", "electrode_support_k", 4, legacy_sections=("field_shape",))

    cfg.score_entropy_weight = _getfloat("benchmarking", "score_entropy_weight", 1.0)
    cfg.score_isolated_weight = _getfloat("benchmarking", "score_isolated_weight", 0.2)
    cfg.score_disconnected_weight = _getfloat("benchmarking", "score_disconnected_weight", 5.0)
    cfg.benchmark_name = _getclean("benchmarking", "benchmark_name", "balanced")
    cfg.benchmark_profile = _getclean("benchmarking", "benchmark_profile", "balanced")
    cfg.save_results = _getbool("optimization", "save_results", True)
    cfg.enforce_connectivity = _getbool("optimization", "enforce_connectivity", True)
    cfg.repair_disconnected = _getbool("optimization", "repair_disconnected", False)

    default_output_dir = os.path.join(os.path.dirname(os.path.abspath(config_file)), "results")
    cfg.output_dir = str(_get("optimization", "output_dir", default_output_dir))
    cfg.run_name = str(_get("optimization", "run_name", "") or "").strip()

    return cfg


def create_default_config(config_file="config.ini"):
    config = configparser.ConfigParser()

    config["combined"] = {
        "seed": "7",
        "h0": "0.1",
        "low_cond": "1000",
        "high_cond": "500",
        "field_model": "fractal",
    }

    config["mesh_generation"] = {
        "n_el": "16",
        "h0": "0.1",
        "model": "fractal",
        "rbf_rows": "4",
        "rbf_cols": "5",
        "rbf_sigma_frac": "0.90",
        "rbf_weight_scale": "0.35",
        "fractal_iter_max": "50",
        "fractal_type": "branching",
        "fractal_power": "3",
        "fractal_shift_frac": "1.0",
        "fractal_threshold_init": "0.00",
        "fractal_center_origin": "true",
        "show_mesh": "true",
        "show_element_ids": "false",
    }

    config["benchmarking"] = {
        "benchmark_name": "balanced",
        "benchmark_profile": "balanced",
        "score_entropy_weight": "1.0",
        "score_isolated_weight": "0.2",
        "score_disconnected_weight": "5.0",
    }

    config["optimization"] = {
        "generations": "100",
        "pop_size": "24",
        "elite_count": "2",
        "crossover_rate": "0.8",
        "mutation_rate": "0.01",
        "tournament_size": "3",
        "optimizer": "de",
        "de_maxiter": "25",
        "de_popsize": "6",
        "de_mutation_min": "0.5",
        "de_mutation_max": "1.0",
        "de_recombination": "0.7",
        "field_model": "fractal",
        "rbf_rows": "4",
        "rbf_cols": "5",
        "rbf_sigma_frac": "0.90",
        "rbf_bias_init": "0.25",
        "rbf_weight_scale": "0.35",
        "fractal_iter_max": "50",
        "fractal_type": "branching",
        "fractal_power": "3",
        "fractal_shift_frac": "1",
        "fractal_threshold_init": "0.00",
        "fractal_center_origin": "true",
        "branching_depth_max": "5",
            "branching_max_children": "2",
        "branching_angle_frac": "0.34",
        "branching_length_decay": "0.68",
        "branching_width_frac": "0.12",
        "branching_width_decay": "0.80",
        "branching_mirror_vertical": "false",
        "branching_aim_electrodes": "false",
        "branching_target_blend": "0.00",
        "branching_root_x_frac": "0.18",
        "branching_root_y_frac": "0.0",
        "branching_seed_all_electrodes": "true",
        "branching_force_meet_center": "true",
        "branching_meet_x_frac": "0.0",
        "branching_meet_y_frac": "0.0",
        "branching_meet_blend": "1.0",
        "branching_random_angle_frac": "0.0",
        "branching_random_center_boost": "2.2",
        "branching_random_center_power": "2.0",
        "branching_force_touch_all_electrodes": "true",
        "electrode_support_k": "4",
        "save_results": "true",
        "output_dir": "results",
        "run_name": "",
        "enforce_connectivity": "true",
        "repair_disconnected": "false",
    }

    with open(config_file, "w", encoding="utf-8") as f:
        config.write(f)


def _build_field_shape_cfg(args, field_geometry):
    cfg = {
        "model": str(args.field_model).strip().lower(),
        "rbf_rows": int(max(1, args.rbf_rows)),
        "rbf_cols": int(max(1, args.rbf_cols)),
        "rbf_sigma_frac": float(args.rbf_sigma_frac),
        "rbf_bias_init": float(args.rbf_bias_init),
        "rbf_weight_scale": float(args.rbf_weight_scale),
        "fractal_iter_max": int(max(1, args.fractal_iter_max)),
        "fractal_type": "branching",
        "fractal_power": int(max(2, args.fractal_power)),
        "fractal_shift_frac": float(max(0.0, args.fractal_shift_frac)),
        "fractal_threshold_init": float(np.clip(args.fractal_threshold_init, -1.0, 1.0)),
        "fractal_center_origin": bool(args.fractal_center_origin),
        "branching_depth_max": int(max(1, args.branching_depth_max)),
            "branching_max_children": int(max(1, args.branching_max_children)),
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
    cfg["branching_electrode_points_norm"] = field_geometry.get("electrode_points_norm")
    cfg["branching_electrode_points_norm_right"] = field_geometry.get("electrode_points_norm_right")
    return cfg


def run(config_file=None):
    if config_file is None:
        config_dir = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(config_dir, "config.ini")

    args = load_config(config_file)

    mesh_obj, protocol_obj, variable_mask, p1, p2 = build_model(h0=args.h0)
    fwd = EITForward(mesh_obj, protocol_obj)

    print(f"Total elements: {mesh_obj.element.shape[0]}")
    print(f"Optimized variable elements: {np.count_nonzero(variable_mask)}")

    adjacency = build_element_adjacency(mesh_obj)
    electrode_supports = build_electrode_support_elements(
        mesh_obj,
        variable_mask,
        k_nearest=args.electrode_support_k,
    )

    field_shape_cfg_seed = {
        "model": str(args.field_model).strip().lower(),
        "rbf_rows": int(max(1, args.rbf_rows)),
        "rbf_cols": int(max(1, args.rbf_cols)),
        "rbf_sigma_frac": float(args.rbf_sigma_frac),
        "fractal_iter_max": int(max(1, args.fractal_iter_max)),
    }
    field_geometry = build_field_geometry(
        mesh_obj=mesh_obj,
        variable_mask=variable_mask,
        p1=p1,
        p2=p2,
        field_shape_cfg=field_shape_cfg_seed,
    )
    field_shape_cfg = _build_field_shape_cfg(args, field_geometry)

    benchmark_weights = get_profile_weights(
        args.benchmark_profile,
        overrides={
            "entropy": float(args.score_entropy_weight),
            "isolated_penalty": float(args.score_isolated_weight),
            "disconnected_penalty": float(args.score_disconnected_weight),
        },
    )
    print(f"Benchmark: {args.benchmark_name}")
    print(f"Benchmark profile: {args.benchmark_profile}")
    print(f"Available benchmarks: {', '.join(list_benchmarks())}")

    if args.optimizer == "de":
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
            enforce_connectivity=args.enforce_connectivity,
            repair_disconnected=args.repair_disconnected,
            benchmark_name=args.benchmark_name,
            benchmark_weights=benchmark_weights,
        )
    else:
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
            tournament_size=args.tournament_size,
            seed=args.seed,
            adjacency=adjacency,
            electrode_supports=electrode_supports,
            enforce_connectivity=args.enforce_connectivity,
            repair_disconnected=args.repair_disconnected,
            benchmark_name=args.benchmark_name,
            benchmark_weights=benchmark_weights,
        )

    best_sens = np.asarray(result["best_sensitivity"]).ravel()
    print(f"Best score found: {result['best_score']:.6e}")
    print(f"Best sensitivity min: {best_sens.min():.6e}")
    print(f"Best sensitivity max: {best_sens.max():.6e}")

    high_elems = result["best_state"]
    is_connected = has_connected_high_elements(high_elems, adjacency)
    electrodes_linked = electrodes_connected_by_high_region(high_elems, adjacency, electrode_supports)
    high_components = get_connected_components(high_elems, adjacency)
    isolated_count = count_isolated_high_elements(high_elems, adjacency)
    print(f"High elements connected: {is_connected}")
    print(f"All electrodes connected by high region: {electrodes_linked}")
    print(f"High connected components: {len(high_components)}")
    print(f"Isolated high elements: {isolated_count}")

    if bool(args.save_results):
        run_name = _build_run_name(args)
        artifact_paths = _save_run_artifacts(
            output_dir=args.output_dir,
            run_name=run_name,
            mesh_obj=mesh_obj,
            p1=p1,
            p2=p2,
            result=result,
            args=args,
            field_shape_cfg=field_shape_cfg,
            benchmark_weights=benchmark_weights,
            connectivity_info={
                "is_connected": is_connected,
                "electrodes_linked": electrodes_linked,
                "components": len(high_components),
                "isolated_count": isolated_count,
            },
        )
        print("Saved run artifacts:")
        print(f"  - NPZ: {artifact_paths['npz']}")
        print(f"  - Metadata: {artifact_paths['metadata_json']}")
        print(f"  - History CSV: {artifact_paths['history_csv']}")
        print(f"  - Element CSV: {artifact_paths['elements_csv']}")

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
        comparison_rows=[],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimization runner.")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to INI config file. Defaults to BendingSim/config.ini.",
    )
    args = parser.parse_args()
    run(config_file=args.config)


if __name__ == "__main__":
    main()
