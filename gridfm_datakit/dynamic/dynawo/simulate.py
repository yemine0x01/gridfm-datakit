"""
Dynaωo simulation functions.

Provides three public functions:

- compute_balanced_static_state_dynawo : OPF → update pypowsybl → AC-PF
- run_dynawo_simulation                : apply mappings and run Dynawo via pypowsybl.dynamic
- _format_dynamic_res                  : convert raw pypowsybl.dynamic output to DynamicResults
"""

from __future__ import annotations

import copy
import time
from typing import Any, Dict, Tuple

import numpy as np

import gridfm_datakit.powsybl as powsybl
from gridfm_datakit.dynamic import DynamicResults
from gridfm_datakit.network import Network
from gridfm_datakit.process.process_network import pf_post_processing, pf_preprocessing
from gridfm_datakit.process.solvers import run_opf

from .api import _get_pypowsybl_dynamic, check_pypowsybl_dynamic_available
from . import DynawoMappings


# ---------------------------------------------------------------------------
# Public: balanced static state
# ---------------------------------------------------------------------------


def compute_balanced_static_state_dynawo(
    pp_net,
    gfm_net: Network,
    julia: Any,
    scenario_index: int = 0,
) -> Tuple[Any, Dict[str, Any]]:
    """Compute the balanced initial conditions for a dynamic simulation.

    Runs the four-step sequence required to produce a consistent initial
    state for Dynaωo:

    1. **OPF** via Julia/PowerModels on the randomised gfm network
       → optimal dispatch for the current scenario.
    2. **update_powsybl** (powsybl submodule)
       → applies OPF results (Pg, Vm setpoints) onto the pypowsybl object
       with correct per-unit conventions.
    3. **AC-PF** via pypowsybl OpenLoadFlow
       → verifies convergence and produces the balanced initial state.
    4. **get_pf_res / pf_post_processing** (powsybl submodule)
       → formats pypowsybl PF results in the gridfm column schema
       with ID-based bus index assignment.

    Parameters
    ----------
    pp_net :
        pypowsybl network (mutated in-place with the balanced state).
        The caller must pass a per-worker *clone/variant* to avoid
        cross-scenario contamination.
    gfm_net : Network
        Randomised gridfm network for the current scenario (already has
        load scenario and perturbations applied).
    julia :
        Initialised Julia interface (from ``init_julia``).
    scenario_index : int
        Used to label the results row (matches ``pf_post_processing``'s
        ``scenario_index`` argument).

    Returns
    -------
    pp_net :
        The updated pypowsybl network, balanced and ready for dynamic
        simulation.
    pf_data : dict
        Power flow results in gridfm column schema with keys:
        ``"bus"``, ``"gen"``, ``"branch"``, ``"Y_bus"``, ``"runtime"``.

    Raises
    ------
    RuntimeError
        If OPF fails to converge.
    ValueError
        If the AC power flow does not converge.
    """
    # Step 1: run OPF on the gfm network to get optimal dispatch
    opf_res = run_opf(gfm_net, julia)

    # Step 2: apply OPF setpoints to gfm_net, then push to pypowsybl
    gfm_net_pf = copy.deepcopy(gfm_net)
    gfm_net_pf = pf_preprocessing(gfm_net_pf, opf_res)

    mapping_p2g = powsybl.build_p2g_maps(gfm_net_pf, pp_net)
    powsybl.update_powsybl(pp_net, gfm_net_pf, mapping_p2g)

    # Step 3: run AC-PF via pypowsybl OpenLoadFlow
    lf_params = powsybl.get_default_lf_params()
    t0 = time.perf_counter()
    pf_metadata = powsybl.pypowsybl.loadflow.run_ac(pp_net, lf_params)
    solve_time = time.perf_counter() - t0

    # Step 4: format results in gridfm column schema (ID-based bus assignment)
    pf_res = powsybl.get_pf_res(pp_net, solve_time, pf_metadata, mapping_p2g)
    pf_data = pf_post_processing(
        scenario_index,
        gfm_net_pf,
        pf_res,
        res_dc=None,
        include_dc_res=False,
    )

    return pp_net, pf_data


# ---------------------------------------------------------------------------
# Public: run Dynawo simulation
# ---------------------------------------------------------------------------


def run_dynawo_simulation(
    pp_net,
    dynamic_mappings: DynawoMappings,
    parameters: Any,
) -> DynamicResults:
    """Apply Dynawo mappings to a balanced pypowsybl network and run the simulation.

    Parameters
    ----------
    pp_net :
        Balanced pypowsybl network (output of ``compute_balanced_static_state_dynawo``).
    dynamic_mappings : DynawoMappings
        Validated Dynawo-ready mapping DataFrames (models, events, curves).
    parameters :
        ``pypowsybl.dynamic.Parameters`` object (from ``prepare_dynawo_parameters``).

    Returns
    -------
    DynamicResults
        Solver-agnostic container with a Zarr-shaped array
        ``(n_variables, n_timesteps)`` and the solver status report string.
    """
    check_pypowsybl_dynamic_available()
    dyn = _get_pypowsybl_dynamic()

    # Build pypowsybl.dynamic mapping objects
    model_mapping = _build_model_mapping(dyn, dynamic_mappings)
    event_mapping = _build_event_mapping(dyn, dynamic_mappings)
    curve_mapping = _build_curve_mapping(dyn, dynamic_mappings)

    # Run simulation
    sim_result = dyn.run_simulation(
        network=pp_net,
        model_mapping=model_mapping,
        event_mapping=event_mapping,
        output_variables_mapping=curve_mapping,
        parameters=parameters,
    )

    return _format_dynamic_res(sim_result, dynamic_mappings)


# ---------------------------------------------------------------------------
# Private: format raw Dynawo output
# ---------------------------------------------------------------------------


def _format_dynamic_res(
    sim_result: Any,
    dynamic_mappings: DynawoMappings,
) -> DynamicResults:
    """Convert raw pypowsybl.dynamic simulation output into DynamicResults.

    The curves from pypowsybl.dynamic are a dict or DataFrame with one entry
    per monitored variable. This function stacks them into a numpy array
    shaped (n_variables, n_timesteps) and wraps it in a Zarr array stored
    in memory (zarr.MemoryStore) so it is ready to be written to a persistent
    Zarr group by _save_generated_data.

    Parameters
    ----------
    sim_result :
        Raw object returned by ``pypowsybl.dynamic.run_simulation``.
    dynamic_mappings : DynawoMappings
        Used to determine expected variable order.

    Returns
    -------
    DynamicResults
    """
    import zarr

    # Extract status report
    status = str(getattr(sim_result, "status", "UNKNOWN"))

    # Extract curves → numpy array (n_variables, n_timesteps)
    curves = getattr(sim_result, "curves", None)

    if curves is None or (hasattr(curves, "empty") and curves.empty):
        # No time-series data (simulation failed or no variables requested)
        data_array = np.empty((0, 0), dtype=np.float64)
    else:
        import pandas as pd

        if isinstance(curves, dict):
            curves = pd.DataFrame(curves)

        # Columns are variable names; rows are timesteps.
        # Drop the "time" column if present.
        if "time" in curves.columns:
            variable_cols = [c for c in curves.columns if c != "time"]
        else:
            variable_cols = list(curves.columns)

        data_array = curves[variable_cols].to_numpy(dtype=np.float64).T
        # shape: (n_variables, n_timesteps)

    # Wrap in a Zarr in-memory array so the caller can write it to a store
    store = zarr.MemoryStore()
    root = zarr.open_group(store, mode="w")
    root.create_dataset(
        "curves",
        data=data_array,
        chunks=(data_array.shape[0], min(data_array.shape[1], 1000))
        if data_array.size > 0
        else True,
        dtype="float64",
        compressor=None,  # compression applied when writing to persistent store
    )

    return DynamicResults(dynamic_results=root["curves"], report=status)


# ---------------------------------------------------------------------------
# Private: pypowsybl.dynamic object builders
# ---------------------------------------------------------------------------


def _build_model_mapping(dyn, dynamic_mappings: DynawoMappings):
    """Build pypowsybl.dynamic ModelMapping from DynawoMappings.dynamic_model_mapping."""
    model_mapping = dyn.ModelMapping()
    for row in dynamic_mappings.dynamic_model_mapping.itertuples(index=False):
        model_mapping.add_all_dynamic_mappings(
            static_id=row.static_id,
            dynamic_model_id=row.dynamic_model_id,
            parameter_set_id=row.parameter_set_id,
        )
    return model_mapping


def _build_event_mapping(dyn, dynamic_mappings: DynawoMappings):
    """Build pypowsybl.dynamic EventMapping from DynawoMappings.event_mapping."""
    event_mapping = dyn.EventMapping()
    for row in dynamic_mappings.event_mapping.itertuples(index=False):
        event_mapping.add_event(
            static_id=row.static_id,
            event_model_id=row.event_model_id,
            parameter_set_id=row.parameter_set_id,
        )
    return event_mapping


def _build_curve_mapping(dyn, dynamic_mappings: DynawoMappings):
    """Build pypowsybl.dynamic CurveMapping from DynawoMappings.variable_mapping."""
    curve_mapping = dyn.CurveMapping()
    for row in dynamic_mappings.variable_mapping.itertuples(index=False):
        curve_mapping.add_curve(
            dynamic_model_id=row.dynamic_model_id,
            variable=row.variable,
        )
    return curve_mapping
