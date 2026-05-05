# Analytical Reconstruction Benchmark Tab

## Overview

The redesigned Benchmark tab provides two distinct workflows:

1. **Analytical Reconstruction Tests** - Comprehensive testing of mesh sensitivity and reconstruction capability
2. **Legacy Quick Benchmark** - Simple scoring of previously saved optimization results

---

## Workflow 1: Analytical Reconstruction Tests

### Purpose
Test the ability of a mesh to:
- Generate accurate sensitivity maps
- Localize touch/contact points at varying distances from the mesh boundary
- Recover geometric shapes and patterns

### Process

#### Step 1: Load Mesh
- **Button:** "Browse Mesh"
  - Opens file dialog to select an NPZ file
  - Must contain: `node`, `element`, `el_pos` arrays
  - Optional: `p1`, `p2` (for bounding box definition)

- **Button:** "Load Mesh"
  - Loads the selected NPZ file
  - Displays mesh statistics (elements, nodes, electrodes)
  - Prepares for analytical testing

#### Step 2: Run Tests
- **Button:** "Run All Analytical Tests"
  - Computes forward problem (Jacobian) for uniform conductivity
  - Generates synthetic anomalies:
    - 3 touch points at distances 0.2, 0.5, 0.8 from boundary
    - Cross shape pattern
    - Star shape pattern (5 arms)
    - C-shape pattern
  - Evaluates each pattern against sensitivity
  - Stores results for reporting

#### Step 3: Generate Report
- **Button:** "Generate Test Report"
  - Prompts to save detailed report as text file
  - Displays report in interactive window
  - Includes metrics and interpretation guide

### Metrics Explained

Each test produces four quality metrics:

1. **Correlation** (−1 to 1, higher is better)
   - Measures alignment between synthetic pattern and sensitivity distribution
   - 1.0 = perfect match, 0.0 = no correlation, −1.0 = inverted

2. **L2 Error** (0 to ∞, lower is better)
   - Root mean squared difference between pattern and sensitivity
   - Normalized difference: 0.0 = perfect match

3. **SNR (dB)** (−∞ to ∞, higher is better)
   - Signal-to-noise ratio of energy in target region
   - 0 dB = target energy equals background energy
   - +10 dB = target is 10× stronger than background

4. **Localization** (0 to 1, higher is better)
   - Fraction of total sensitivity concentrated in target region
   - 1.0 = all sensitivity in target, 0.0 = none in target

### Interpreting Results

**Good mesh properties:**
- Touch points: High correlation (>0.5), high SNR (>10 dB), high localization (>0.3)
- Shapes: Correlation >0.4, SNR >5 dB, localization >0.2
- Low L2 error across all tests (<0.3)

**Poor mesh properties:**
- Low correlation (<0.2) indicates sensitivity doesn't match pattern
- Low SNR (<0 dB) indicates target not distinguishable from background
- Low localization (<0.1) indicates sensitivity spread diffusely

---

## Workflow 2: Legacy Quick Benchmark

### Purpose
Score the conductivity state from a previously saved optimization run using standard benchmarks.

### Process

1. **Browse NPZ**
   - Select a saved optimization result NPZ file
   
2. **Score Saved Result**
   - Evaluates conductivity state with selected benchmark
   - Displays: score, connectivity status, isolated elements count
   - Shows source file path

### Benchmarks Available
- **Uniformity** - Minimize sensitivity variance
- **Expected Sensitivity** - Touch patch average
- **Minimax Sensitivity** - Touch patch minimum
- **Softmin Sensitivity** - Soft minimum with temperature parameter
- **SNR Sensitivity** - Signal-to-noise in touch regions
- **Distinguishability** - Pattern distinction capability
- **Combined** - Mixed objectives

---

## File Formats

### Input: Mesh NPZ
Required arrays:
```
node        : (n_nodes, 2) - node coordinates
element     : (n_elements, 3) - triangles (vertex indices)
el_pos      : (n_electrodes,) - electrode positions (node indices)
```

Optional arrays:
```
p1, p2      : bounding box corners for distance calculations
```

### Output: Report TXT
Human-readable format with:
- Mesh statistics
- Test results for each synthetic pattern
- Quality metrics
- Interpretation guide
- Timestamp and source file

---

## Advanced: Customizing Shape Tests

Edit `BendingSim/analytical_reconstruction.py` to customize:

- `touch_point_at_distance_from_edge()` - Change distance fractions or position
- `shape_cross()` - Adjust thickness, orientation
- `shape_star()` - Modify number of arms, radius, thickness
- `shape_c()` - Change radius, opening angle
- `run_all_reconstructions()` - Add new tests or change parameters

Example: Test star with 6 arms instead of 5:
```python
star_state = shape_star(
    mesh_centroids, bbox,
    n_arms=6,  # Changed from 5
    radius_frac=0.3
)
```

---

## Troubleshooting

**"No mesh loaded"**
- Click "Browse Mesh" to select an NPZ file
- Click "Load Mesh" to validate and load it

**"Tests failed"**
- Ensure mesh NPZ has required arrays: node, element, el_pos
- Check mesh has reasonable dimensions (not degenerate)
- Verify electrode count >= 4

**Missing report metrics**
- Run "Run All Analytical Tests" first
- Check that tests completed without errors
- Verify mesh loaded successfully

**Poor test results**
- Very coarse meshes (<100 elements) may have poor localization
- Meshes with irregular element sizes may show variable results
- Check electrode count is adequate (typically >=16)

