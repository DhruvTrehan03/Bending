# Refactoring Summary: Simplified DE UI with Element-Wise Conductivity

## Overview
`combined_optimizer_ui.py` has been refactored to focus exclusively on **DE (Differential Evolution) optimization** with **individual element conductivity states** as parameters and **uniform sensitivity** (jacobian-based) as the only objective.

## Major Changes

### 1. **Removed Field Parameterization**
- **Deleted**: All RBF, Fractal, and Branching field model UI controls
  - `field_model_var`, `rbf_frame`, `fractal_frame`, `branching_frame` removed
  - `rbf_rows_var`, `rbf_cols_var`, `rbf_sigma_frac_var`, etc. removed
  - `fractal_iter_max_var`, `fractal_power_var`, `branching_depth_max_var`, etc. removed
  
- **Rationale**: Each parameter now directly corresponds to one variable element's conductivity state (binary: high/low), eliminating the need for field basis functions.

### 2. **Removed GA Optimizer**
- **Deleted**: All GA-related UI controls and logic
  - `ga_generations_var`, `ga_pop_size_var`, `ga_elite_count_var`, etc. removed
  - `ga_crossover_var`, `ga_mutation_var`, `ga_tournament_var` removed
  - GA optimization code path removed from `start_optimization()`
  
- **Deleted**: `optimize_parameterized_field_ga` import
  
- **Rationale**: DE is more suitable for binary element state optimization.

### 3. **Simplified Objective Selection**
- **Deleted**: Multiple benchmark profiles
  - `optimization_goal_var`, `benchmark_profile_post_var` removed
  - `score_entropy_var`, `score_isolated_var`, `score_disconnected_var` removed
  - `list_benchmarks()` and `get_profile_weights()` imports removed
  
- **New**: Fixed single objective: **`sensitivity_uniformity`**
  - Objective: Minimize variance in jacobian-based sensitivity values
  - New parameter: `sens_variance_weight_var` (weight for variance penalty)
  - Optional: `sens_mean_target_var` (auto-calculated or user-specified mean)

### 4. **Updated Imports**
```python
# REMOVED:
- build_field_geometry
- parameter_vector_size_from_cfg
- get_profile_weights
- list_benchmarks
- optimize_parameterized_field_ga

# KEPT:
- build_element_adjacency
- build_electrode_support_elements
- build_model
- benchmark_saved_result
- has_connected_high_elements
- electrodes_connected_by_high_region
- plot_results
- optimize_parameterized_field_de
```

### 5. **Simplified OptimizationContext Dataclass**
Removed fields:
- `field_shape_cfg` - no longer needed (element model has no configuration)
- `optimizer` - always "de"

### 6. **Updated UI Tabs**

#### Mesh Tab
- Same as before (generate rectangle mesh, set electrode count)

#### Optimization Tab (SIMPLIFIED)
- **New layout**: Sensitivity Uniformity Objective section
  - `sens_mean_target_var`: sensitivity mean target ("auto" or numeric)
  - `sens_variance_weight_var`: variance penalty weight
  - `enforce_connectivity_var`: enforce electrode connectivity (optional)
  
- **DE Settings section** (condensed)
  - `low_cond_var`, `high_cond_var` (conductivity range)
  - `seed_var` (random seed)
  - DE Parameters:
    - `de_maxiter_var`, `de_popsize_var`
    - `de_mutation_min_var`, `de_mutation_max_var`
    - `de_recombination_var`
    - `de_seed_var` (DE-specific seed)

#### Save Tab
- Unchanged (save optimization artifacts)

#### Benchmark Tab (SIMPLIFIED)
- Removed profile selection
- Now scores saved NPZ files using fixed `sensitivity_uniformity` objective
- Only needs NPZ path + optional JSON output path

### 7. **Parameter Count**
- **Old**: Variable based on field parameterization (RBF: 1 + rows×cols; Fractal: 5-7)
- **New**: Always equals number of variable elements in mesh
  - For 16-element circle: typically 13-15 variable elements = 13-15 parameters
  - Display: "Parameters: N (one per variable element)"

### 8. **Optimization Workflow Changes**

**start_optimization()**:
```python
# OLD: Chose GA vs DE based on optimizer_var
# NEW: Always uses DE via optimize_parameterized_field_de()

# Parameters:
- field_geometry=None  # Not used for element model
- field_shape_cfg={"model": "element"}  # Direct element states
- benchmark_name="sensitivity_uniformity"  # Fixed
- benchmark_weights={"sensitivity_variance": weight}  # Fixed
```

### 9. **Element Model Details**
- Each parameter: `theta[i]` ∈ [0, 1] for variable element i
- Conversion: `theta[i] > 0.5` → high conductivity, else low conductivity (or direct threshold)
- Connectivity: Enforced via `has_connected_high_elements()` if `enforce_connectivity_var=True`

### 10. **Removed Methods**
- `_update_parameter_visibility()` - no longer needed (no field model switching)
- `_build_field_shape_cfg()` - no longer needed

### 11. **Modified Methods**

**_build_context()**:
- Removed `field_shape_cfg` construction
- Removed `field_geometry` building
- `benchmark_name` always "sensitivity_uniformity"
- `benchmark_weights` only has "sensitivity_variance" key

**_build_save_args()**:
- `optimizer` always "de"
- `benchmark_name` always "sensitivity_uniformity"

**_current_run_name()**:
- Simplified: uses "de" + "sensitivity_uniformity" as defaults

**save_current_results()**:
- `field_shape_cfg={"model": "element"}` (fixed)

**open_detailed_result_plot()**:
- `field_shape_cfg={"model": "element"}` (fixed)

**run_saved_benchmark()**:
- Always uses "sensitivity_uniformity" (no profile selection)
- Only `sens_variance_weight_var` is available for override

## UI Title
- **Old**: "BendingSim Combined Mesh + Optimization UI"
- **New**: "BendingSim DE Uniform Sensitivity Optimizer"

## Testing Checklist
- [ ] Mesh generation works
- [ ] Parameter count displays correctly (# variable elements)
- [ ] DE optimization runs successfully
- [ ] Live score updates display during optimization
- [ ] Best/current mesh renders correctly
- [ ] Sensitivity variance score calculated properly
- [ ] Save artifacts works
- [ ] Benchmark saved results with sensitivity_uniformity
- [ ] No references to removed variables/methods

## Future Enhancements
- Add sensitivity uniformity metric display (mean, std dev, min/max)
- Add optional regularization (e.g., penalize too few high-conductivity elements)
- Support different connectivity constraints (not just electrode-to-electrode)
