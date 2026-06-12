"""
Tests for run_dynawo_simulation and _format_dynamic_res.

All tests that call into pypowsybl.dynamic are skipped when the dynamic
extras are not installed.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from gridfm_datakit.dynamic import DynamicResults


# ---------------------------------------------------------------------------
# _format_dynamic_res — unit-testable without pypowsybl.dynamic
# ---------------------------------------------------------------------------


class TestFormatDynamicRes:
    """_format_dynamic_res can be tested by passing a mock sim_result."""

    def _make_mappings(self):
        from gridfm_datakit.dynamic.dynawo import DynawoMappings

        return DynawoMappings(
            dynamic_model_mapping=pd.DataFrame(
                {
                    "static_id": ["G1"],
                    "dynamic_model_id": ["M1"],
                    "parameter_set_id": ["p0"],
                },
            ),
            event_mapping=pd.DataFrame(
                {
                    "static_id": ["L1"],
                    "event_model_id": ["Ev"],
                    "parameter_set_id": ["e0"],
                },
            ),
            variable_mapping=pd.DataFrame(
                {"dynamic_model_id": ["G1"], "variable": ["omega"]},
            ),
        )

    def _make_sim_result(self, n_vars: int = 2, n_t: int = 50) -> MagicMock:
        times = np.linspace(0, 10, n_t)
        data = {"time": times}
        for i in range(n_vars):
            data[f"var_{i}"] = np.sin(times + i)
        result = MagicMock()
        result.status = "CONVERGED"
        result.curves = pd.DataFrame(data)
        return result

    def test_returns_dynamic_results(self):
        from gridfm_datakit.dynamic.dynawo.simulate import _format_dynamic_res

        sim_result = self._make_sim_result(n_vars=2, n_t=50)
        mappings = self._make_mappings()
        dr = _format_dynamic_res(sim_result, mappings)
        assert isinstance(dr, DynamicResults)

    def test_report_is_string(self):
        from gridfm_datakit.dynamic.dynawo.simulate import _format_dynamic_res

        sim_result = self._make_sim_result()
        mappings = self._make_mappings()
        dr = _format_dynamic_res(sim_result, mappings)
        assert isinstance(dr.report, str)

    def test_schema_shape(self):
        """dynamic_results array must have shape (n_variables, n_timesteps)."""
        from gridfm_datakit.dynamic.dynawo.simulate import _format_dynamic_res

        N_VARS, N_T = 3, 100
        sim_result = self._make_sim_result(n_vars=N_VARS, n_t=N_T)
        mappings = self._make_mappings()
        dr = _format_dynamic_res(sim_result, mappings)
        arr = np.array(dr.dynamic_results)
        assert arr.shape == (N_VARS, N_T), (
            f"Expected shape ({N_VARS}, {N_T}), got {arr.shape}"
        )

    def test_empty_curves_returns_empty_array(self):
        """When sim_result.curves is None, array must be empty — no exception."""
        from gridfm_datakit.dynamic.dynawo.simulate import _format_dynamic_res

        sim_result = MagicMock()
        sim_result.status = "FAILED"
        sim_result.curves = None
        mappings = self._make_mappings()
        dr = _format_dynamic_res(sim_result, mappings)
        arr = np.array(dr.dynamic_results)
        assert arr.size == 0

    def test_status_propagated(self):
        from gridfm_datakit.dynamic.dynawo.simulate import _format_dynamic_res

        sim_result = self._make_sim_result()
        sim_result.status = "CONVERGED"
        dr = _format_dynamic_res(sim_result, self._make_mappings())
        assert "CONVERGED" in dr.report


class TestReturnsDynamicResults:
    """Integration-style tests — skipped when pypowsybl.dynamic is absent."""

    def test_run_dynawo_simulation_skipped_without_dynamic(
        self,
        sample_dynawo_mappings,
        pp_net_ieee14,
    ):
        """run_dynawo_simulation raises ImportError (not AttributeError) when
        pypowsybl.dynamic is unavailable — import guard is working."""
        from gridfm_datakit.dynamic.dynawo.api import is_pypowsybl_dynamic_available

        if is_pypowsybl_dynamic_available():
            pytest.skip("pypowsybl.dynamic is available; use the full integration test")

        from gridfm_datakit.dynamic.dynawo.simulate import run_dynawo_simulation

        with pytest.raises(ImportError, match="pypowsybl dynamic simulation support"):
            run_dynawo_simulation(pp_net_ieee14, sample_dynawo_mappings, object())
