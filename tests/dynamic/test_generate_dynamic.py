"""
Tests for generate_dynamic_data — config validation and entry point.

Full end-to-end tests require pypowsybl + Julia + Dynawo and are skipped
when unavailable. Config validation tests are always runnable.
"""

from __future__ import annotations

import tempfile

import pandas as pd
import pytest

from gridfm_datakit.dynamic.generate_dynamic import (
    _load_and_prep_dynamic_mappings,
    _validate_dynamic_config,
)
from gridfm_datakit.utils.param_handler import NestedNamespace


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestRaisesOnWrongSource:
    def test_raises_when_source_not_powsybl(self):
        """generate_dynamic_data must raise ValueError if source != 'powsybl'."""
        config = NestedNamespace(
            network=NestedNamespace(
                name="case14_ieee",
                source="pglib",
                reader="native",
            ),
            load=NestedNamespace(generator="agg_load_profile", scenarios=5),
            topology_perturbation=NestedNamespace(type="random", k=0),
            generation_perturbation=NestedNamespace(type="none"),
            admittance_perturbation=NestedNamespace(type="none"),
            settings=NestedNamespace(
                mode="pf",
                num_processes=1,
                data_dir=tempfile.mkdtemp(),
                seed=42,
                overwrite=True,
                max_iter=200,
                large_chunk_size=10,
                include_dc_res=False,
                pf_fast=False,
                dcpf_fast=False,
                enable_solver_logs=False,
                pf_solver="powsybl",
            ),
            dynamic=NestedNamespace(
                dynamic_solver="dynawo",
                dynamic_models_file="models.csv",
                events_file="events.csv",
                variables_file="variables.csv",
                output_dir=tempfile.mkdtemp(),
                solver_parameters=NestedNamespace(
                    solver_type="SIM",
                    start_time=0.0,
                    stop_time=10.0,
                    precision=1e-6,
                ),
            ),
        )
        # source is "pglib", not "powsybl"
        with pytest.raises(ValueError, match="powsybl"):
            _validate_dynamic_config(config)

    def test_raises_when_no_dynamic_block(self):
        config = NestedNamespace(
            network=NestedNamespace(name="case14", source="powsybl"),
        )
        with pytest.raises(ValueError, match="dynamic"):
            _validate_dynamic_config(config)

    def test_raises_when_no_dynamic_solver(self):
        config = NestedNamespace(
            network=NestedNamespace(name="case14", source="powsybl"),
            dynamic=NestedNamespace(dynamic_solver=None),
        )
        with pytest.raises(ValueError, match="dynamic_solver"):
            _validate_dynamic_config(config)

    def test_valid_config_does_not_raise(self):
        config = NestedNamespace(
            network=NestedNamespace(name="case14", source="powsybl", reader="powsybl"),
            dynamic=NestedNamespace(dynamic_solver="dynawo"),
        )
        _validate_dynamic_config(config)  # must not raise


# ---------------------------------------------------------------------------
# _load_and_prep_dynamic_mappings — unknown solver raises NotImplementedError
# ---------------------------------------------------------------------------


class TestLoadAndPrepDynamicMappings:
    def test_unknown_solver_raises(self, tmp_path):
        models_df = pd.DataFrame(
            {
                "static_id": ["G1"],
                "dynamic_model_id": ["Model"],
                "parameter_set_id": ["p0"],
            },
        )
        events_df = pd.DataFrame(
            {
                "static_id": ["L1"],
                "event_model_id": ["EvModel"],
                "parameter_set_id": ["e0"],
            },
        )
        variables_df = pd.DataFrame({"dynamic_model_id": ["G1"], "variable": ["v"]})

        models_path = str(tmp_path / "models.csv")
        events_path = str(tmp_path / "events.csv")
        variables_path = str(tmp_path / "variables.csv")
        models_df.to_csv(models_path, index=False)
        events_df.to_csv(events_path, index=False)
        variables_df.to_csv(variables_path, index=False)

        config = NestedNamespace(
            dynamic=NestedNamespace(
                dynamic_solver="psse_future",  # not implemented
                dynamic_models_file=models_path,
                events_file=events_path,
                variables_file=variables_path,
            ),
        )
        with pytest.raises(NotImplementedError, match="psse_future"):
            _load_and_prep_dynamic_mappings(config)
