#!/usr/bin/env python3
"""Test script to verify element model works correctly with connectivity constraints."""

import sys
import numpy as np
from mesh_generation import (
    build_model,
    build_field_geometry,
    build_element_adjacency,
    build_electrode_support_elements,
    parameter_vector_size_from_cfg,
    parameterized_state_from_theta,
)
from benchmarking import (
    expand_state_to_full_mesh,
    get_state_for_connectivity,
    validate_high_element_connectivity,
    electrodes_connected_by_high_region,
)


def test_element_model_basic():
    """Test basic element model functionality."""
    print("=" * 60)
    print("Test 1: Basic Element Model Functionality")
    print("=" * 60)
    
    # Build mesh and model
    mesh_obj, protocol_obj, variable_mask, p1, p2 = build_model(n_el=16, h0=0.1)
    n_total = mesh_obj.element.shape[0]
    n_variable = np.sum(variable_mask)
    print(f"Total elements: {n_total}")
    print(f"Variable elements: {n_variable}")
    
    # Build field geometry for element model
    field_shape_cfg = {
        "model": "element",
        "rbf_rows": 4,
        "rbf_cols": 5,
        "fractal_type": "branching",
    }
    field_geometry = build_field_geometry(mesh_obj, variable_mask, p1, p2, field_shape_cfg)
    print(f"Field geometry keys: {sorted(field_geometry.keys())}")
    print(f"n_variable_elements: {field_geometry.get('n_variable_elements', 'NOT SET')}")
    
    # Get parameter vector size
    param_size = parameter_vector_size_from_cfg(field_shape_cfg, field_geometry)
    print(f"Parameter vector size: {param_size}")
    assert param_size == n_variable, f"Expected {n_variable}, got {param_size}"
    
    # Test parameterized_state_from_theta
    theta = np.random.randn(param_size)
    state_var, field = parameterized_state_from_theta(theta, field_geometry, field_shape_cfg)
    print(f"State (variable-indexed) size: {state_var.size}")
    print(f"Field size: {field.size}")
    assert state_var.size == n_variable, f"Expected state size {n_variable}, got {state_var.size}"
    print("✓ Parameter-to-state conversion works correctly\n")


def test_state_expansion():
    """Test state expansion from variable-element space to full-mesh space."""
    print("=" * 60)
    print("Test 2: State Expansion to Full Mesh")
    print("=" * 60)
    
    mesh_obj, protocol_obj, variable_mask, p1, p2 = build_model(n_el=16, h0=0.1)
    n_total = mesh_obj.element.shape[0]
    n_variable = np.sum(variable_mask)
    
    # Create a simple state for variable elements
    state_var = np.random.rand(n_variable) > 0.5
    
    # Expand to full mesh
    state_full = expand_state_to_full_mesh(state_var, variable_mask)
    print(f"Variable element state size: {state_var.size}")
    print(f"Full mesh state size: {state_full.size}")
    print(f"Expected total elements: {n_total}")
    assert state_full.size == n_total, f"Expected {n_total}, got {state_full.size}"
    
    # Check that non-variable elements are False
    non_variable_indices = np.where(~variable_mask)[0]
    if non_variable_indices.size > 0:
        assert not np.any(state_full[non_variable_indices]), "Non-variable elements should be False"
        print(f"✓ Non-variable elements are correctly False ({len(non_variable_indices)} elements)")
    
    # Check that variable elements match the input
    variable_indices = np.where(variable_mask)[0]
    np.testing.assert_array_equal(state_full[variable_indices], state_var)
    print("✓ Variable elements match the input state\n")


def test_get_state_for_connectivity():
    """Test automatic state sizing for connectivity functions."""
    print("=" * 60)
    print("Test 3: Automatic State Sizing for Connectivity")
    print("=" * 60)
    
    mesh_obj, protocol_obj, variable_mask, p1, p2 = build_model(n_el=16, h0=0.1)
    n_total = mesh_obj.element.shape[0]
    n_variable = np.sum(variable_mask)
    
    # Test with variable-element-sized state
    state_var = np.random.rand(n_variable) > 0.5
    state_full = get_state_for_connectivity(state_var, variable_mask)
    assert state_full.size == n_total, f"Expected {n_total}, got {state_full.size}"
    print(f"✓ Variable-sized state expanded to full-mesh size ({n_total})")
    
    # Test with full-mesh-sized state
    state_full_input = np.random.rand(n_total) > 0.5
    state_result = get_state_for_connectivity(state_full_input, variable_mask)
    np.testing.assert_array_equal(state_result, state_full_input)
    print(f"✓ Full-mesh-sized state passed through unchanged")
    
    print()


def test_connectivity_validation():
    """Test connectivity validation with element model."""
    print("=" * 60)
    print("Test 4: Connectivity Validation with Element Model")
    print("=" * 60)
    
    mesh_obj, protocol_obj, variable_mask, p1, p2 = build_model(n_el=16, h0=0.1)
    n_variable = np.sum(variable_mask)
    
    # Build adjacency and electrode supports
    adjacency = build_element_adjacency(mesh_obj)
    electrode_supports = build_electrode_support_elements(mesh_obj, variable_mask, k_nearest=4)
    
    # Test 1: All elements high (should be connected)
    state_var = np.ones(n_variable, dtype=bool)
    result = validate_high_element_connectivity(
        state_var, adjacency, 
        electrode_supports=electrode_supports, 
        strict=True, 
        variable_mask=variable_mask
    )
    print(f"All elements high: {result['reason']}")
    print(f"  - Valid: {result['valid']}")
    print(f"  - Components: {result['n_components']}")
    print(f"  - Isolated: {result['isolated_count']}")
    print(f"  - Electrodes connected: {result['all_electrodes_connected']}")
    assert result['valid'], "All elements should be valid"
    print("✓ All high elements form a valid connected region\n")
    
    # Test 2: No high elements (should be invalid)
    state_var = np.zeros(n_variable, dtype=bool)
    result = validate_high_element_connectivity(
        state_var, adjacency, 
        electrode_supports=electrode_supports, 
        strict=True, 
        variable_mask=variable_mask
    )
    print(f"No elements high: {result['reason']}")
    print(f"  - Valid: {result['valid']}")
    assert not result['valid'], "No elements should be invalid"
    print("✓ No high elements correctly marked as invalid\n")
    
    # Test 3: Random connected subset (variable result)
    state_var = np.random.rand(n_variable) > 0.7
    if np.sum(state_var) > 0:
        result = validate_high_element_connectivity(
            state_var, adjacency, 
            electrode_supports=electrode_supports, 
            strict=True, 
            variable_mask=variable_mask
        )
        print(f"Random subset ({np.sum(state_var)} elements): {result['reason']}")
        print(f"  - Valid: {result['valid']}")
        print(f"  - Components: {result['n_components']}")
        print(f"  - Isolated: {result['isolated_count']}")
        print(f"  - Electrodes connected: {result['all_electrodes_connected']}")
        print()


def test_end_to_end():
    """Test end-to-end element model with optimization setup."""
    print("=" * 60)
    print("Test 5: End-to-End Element Model Setup")
    print("=" * 60)
    
    # Build complete setup
    mesh_obj, protocol_obj, variable_mask, p1, p2 = build_model(n_el=16, h0=0.1)
    n_variable = np.sum(variable_mask)
    
    # Field parameterization
    field_shape_cfg = {"model": "element"}
    field_geometry = build_field_geometry(mesh_obj, variable_mask, p1, p2, field_shape_cfg)
    
    # Connectivity setup
    adjacency = build_element_adjacency(mesh_obj)
    electrode_supports = build_electrode_support_elements(mesh_obj, variable_mask, k_nearest=4)
    
    # Generate a parameter vector
    theta = np.random.randn(n_variable)
    
    # Convert to state
    state_var, field = parameterized_state_from_theta(theta, field_geometry, field_shape_cfg)
    print(f"Generated state from {n_variable} parameters")
    print(f"  - State (variable) size: {state_var.size}")
    print(f"  - High elements: {np.sum(state_var)}")
    
    # Convert for connectivity checking
    state_full = get_state_for_connectivity(state_var, variable_mask)
    print(f"  - Expanded to full-mesh: {state_full.size} elements")
    
    # Check connectivity
    is_connected = electrodes_connected_by_high_region(state_full, adjacency, electrode_supports)
    print(f"  - All electrodes connected: {is_connected}")
    
    # Validate
    validation = validate_high_element_connectivity(
        state_var, adjacency, 
        electrode_supports=electrode_supports,
        variable_mask=variable_mask
    )
    print(f"  - Validation result: {validation['reason']}")
    print()


if __name__ == "__main__":
    try:
        test_element_model_basic()
        test_state_expansion()
        test_get_state_for_connectivity()
        test_connectivity_validation()
        test_end_to_end()
        
        print("=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
