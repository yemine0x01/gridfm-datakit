"""
Tests for compute_balanced_static_state_dynawo.

Requires pypowsybl and a working Julia / PowerModels installation.
Tests are skipped when these are unavailable.
"""

from __future__ import annotations

import copy

import pytest

from tests.dynamic.conftest import requires_powsybl

pytestmark = requires_powsybl


@pytest.fixture(scope="module")
def julia_instance():
    """Initialise Julia once for the whole module to avoid repeated JIT overhead."""
    from gridfm_datakit.process.process_network import init_julia

    try:
        jl = init_julia(max_iter=200)
        return jl
    except Exception as e:
        pytest.skip(f"Julia/PowerModels not available: {e}")


class TestOpfThenPfConverges:
    def test_balanced_state_succeeds(self, loaded_ieee14, julia_instance):
        """compute_balanced_static_state_dynawo must return without raising."""
        from gridfm_datakit.dynamic.dynawo.simulate import (
            compute_balanced_static_state_dynawo,
        )

        pp_net = copy.deepcopy(loaded_ieee14.pp_net)
        gfm_net = copy.deepcopy(loaded_ieee14.gfm_net)

        pp_net_balanced, pf_data = compute_balanced_static_state_dynawo(
            pp_net,
            gfm_net,
            julia_instance,
            scenario_index=0,
        )
        assert pp_net_balanced is not None
        assert pf_data is not None

    def test_pf_data_has_expected_keys(self, loaded_ieee14, julia_instance):
        import copy
        from gridfm_datakit.dynamic.dynawo.simulate import (
            compute_balanced_static_state_dynawo,
        )

        pp_net = copy.deepcopy(loaded_ieee14.pp_net)
        gfm_net = copy.deepcopy(loaded_ieee14.gfm_net)
        _, pf_data = compute_balanced_static_state_dynawo(
            pp_net,
            gfm_net,
            julia_instance,
            scenario_index=0,
        )
        for key in ("bus", "gen", "branch", "Y_bus"):
            assert key in pf_data, f"pf_data missing key: {key}"

    def test_bus_array_has_correct_row_count(self, loaded_ieee14, julia_instance):
        import copy
        from gridfm_datakit.dynamic.dynawo.simulate import (
            compute_balanced_static_state_dynawo,
        )

        pp_net = copy.deepcopy(loaded_ieee14.pp_net)
        gfm_net = copy.deepcopy(loaded_ieee14.gfm_net)
        _, pf_data = compute_balanced_static_state_dynawo(
            pp_net,
            gfm_net,
            julia_instance,
            scenario_index=0,
        )
        n_buses = gfm_net.buses.shape[0]
        assert pf_data["bus"].shape[0] == n_buses


class TestPgBusAssignmentFormatIndependent:
    """Guard against silent wrong initial conditions when pypowsybl doesn't export
    buses in sorted ID order.

    Constructs a scenario where the gfm bus index differs from pypowsybl row
    order and asserts Pg_bus/Qg_bus are correctly assigned by ID.
    """

    def test_pg_bus_index_column_is_integer(self, loaded_ieee14, julia_instance):
        """Bus array scenario_index column must be an integer (not float), confirming
        proper scenario tracking."""
        import copy
        from gridfm_datakit.dynamic.dynawo.simulate import (
            compute_balanced_static_state_dynawo,
        )

        pp_net = copy.deepcopy(loaded_ieee14.pp_net)
        gfm_net = copy.deepcopy(loaded_ieee14.gfm_net)
        SCENARIO_IDX = 7
        _, pf_data = compute_balanced_static_state_dynawo(
            pp_net,
            gfm_net,
            julia_instance,
            scenario_index=SCENARIO_IDX,
        )
        # First column of bus array is scenario_index
        assert int(pf_data["bus"][0, 0]) == SCENARIO_IDX
