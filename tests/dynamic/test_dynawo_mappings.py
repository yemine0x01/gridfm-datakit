"""
Tests for DynawoMappings, generate_dynawo_mappings, and validate().
"""

from __future__ import annotations

import pandas as pd
import pytest

from gridfm_datakit.dynamic import DynamicInputs
from gridfm_datakit.dynamic.dynawo import (
    DynawoMappings,
    generate_dynawo_mappings,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dynamic_inputs(
    n_gens: int = 2,
    n_events: int = 1,
    n_vars: int = 1,
) -> DynamicInputs:
    gen_ids = [f"GEN_{i}" for i in range(n_gens)]
    return DynamicInputs(
        dynamic_models=pd.DataFrame(
            {
                "static_id": gen_ids,
                "dynamic_model_id": ["GeneratorSynchronousThreeWindings"] * n_gens,
                "parameter_set_id": [f"p{i}" for i in range(n_gens)],
            },
        ),
        events=pd.DataFrame(
            {
                "static_id": [f"LINE_{i}" for i in range(n_events)],
                "event_model_id": ["EventQuadripoleDisconnection"] * n_events,
                "parameter_set_id": [f"ev{i}" for i in range(n_events)],
            },
        ),
        variables=pd.DataFrame(
            {
                "dynamic_model_id": gen_ids[:n_vars],
                "variable": ["generator_omegaPu"] * n_vars,
            },
        ),
    )


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------


class TestGenerateDynawoMappingsShape:
    def test_returns_dynawo_mappings(self):
        inputs = _make_dynamic_inputs()
        result = generate_dynawo_mappings(inputs)
        assert isinstance(result, DynawoMappings)

    def test_dynamic_model_mapping_rows(self):
        inputs = _make_dynamic_inputs(n_gens=3)
        result = generate_dynawo_mappings(inputs)
        assert len(result.dynamic_model_mapping) == 3

    def test_event_mapping_rows(self):
        inputs = _make_dynamic_inputs(n_events=2)
        result = generate_dynawo_mappings(inputs)
        assert len(result.event_mapping) == 2

    def test_variable_mapping_rows(self):
        inputs = _make_dynamic_inputs(n_vars=2)
        result = generate_dynawo_mappings(inputs)
        assert len(result.variable_mapping) == 2

    def test_dynamic_model_mapping_columns(self):
        inputs = _make_dynamic_inputs()
        result = generate_dynawo_mappings(inputs)
        for col in ("static_id", "dynamic_model_id", "parameter_set_id"):
            assert col in result.dynamic_model_mapping.columns

    def test_event_mapping_columns(self):
        inputs = _make_dynamic_inputs()
        result = generate_dynawo_mappings(inputs)
        for col in ("static_id", "event_model_id", "parameter_set_id"):
            assert col in result.event_mapping.columns

    def test_variable_mapping_columns(self):
        inputs = _make_dynamic_inputs()
        result = generate_dynawo_mappings(inputs)
        for col in ("dynamic_model_id", "variable"):
            assert col in result.variable_mapping.columns


# ---------------------------------------------------------------------------
# validate() tests
# ---------------------------------------------------------------------------


class TestDynawoMappingsValidate:
    def test_valid_mappings_do_not_raise(self):
        inputs = _make_dynamic_inputs()
        mappings = generate_dynawo_mappings(inputs)
        mappings.validate()  # must not raise

    def test_validate_raises_on_missing_model_column(self):
        inputs = _make_dynamic_inputs()
        mappings = generate_dynawo_mappings(inputs)
        # Remove a required column
        mappings.dynamic_model_mapping = mappings.dynamic_model_mapping.drop(
            columns=["parameter_set_id"],
        )
        with pytest.raises(ValueError, match="parameter_set_id"):
            mappings.validate()

    def test_validate_raises_on_missing_event_column(self):
        inputs = _make_dynamic_inputs()
        mappings = generate_dynawo_mappings(inputs)
        mappings.event_mapping = mappings.event_mapping.drop(columns=["event_model_id"])
        with pytest.raises(ValueError, match="event_model_id"):
            mappings.validate()

    def test_validate_raises_on_missing_variable_column(self):
        inputs = _make_dynamic_inputs()
        mappings = generate_dynawo_mappings(inputs)
        mappings.variable_mapping = mappings.variable_mapping.drop(columns=["variable"])
        with pytest.raises(ValueError, match="variable"):
            mappings.validate()


# ---------------------------------------------------------------------------
# ID preservation test
# ---------------------------------------------------------------------------


class TestElementIdsPreservedFromInputs:
    def test_static_ids_match_inputs(self):
        """IDs in DynawoMappings must be exactly those in DynamicInputs (no silent remapping)."""
        inputs = _make_dynamic_inputs(n_gens=3)
        mappings = generate_dynawo_mappings(inputs)
        assert list(mappings.dynamic_model_mapping["static_id"]) == list(
            inputs.dynamic_models["static_id"],
        )

    def test_event_ids_match_inputs(self):
        inputs = _make_dynamic_inputs(n_events=2)
        mappings = generate_dynawo_mappings(inputs)
        assert list(mappings.event_mapping["static_id"]) == list(
            inputs.events["static_id"],
        )

    def test_variable_ids_match_inputs(self):
        inputs = _make_dynamic_inputs(n_vars=2)
        mappings = generate_dynawo_mappings(inputs)
        assert list(mappings.variable_mapping["dynamic_model_id"]) == list(
            inputs.variables["dynamic_model_id"],
        )
