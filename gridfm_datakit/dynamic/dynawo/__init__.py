"""
Dynaωo-specific submodule for gridfm_datakit dynamic simulation.

Exposes:
- DynawoMappings : dataclass holding the three DataFrames that map directly
  onto pypowsybl.dynamic's add_all_dynamic_mappings / event / curve APIs.
- generate_dynawo_mappings : convert a DynamicInputs into DynawoMappings.
- prepare_dynawo_parameters : build a pypowsybl.dynamic.Parameters object
  from the config's dynamic.solver_parameters block.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd

from gridfm_datakit.dynamic import DynamicInputs

from .api import _get_pypowsybl_dynamic, check_pypowsybl_dynamic_available

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Required columns for DynawoMappings DataFrames
# (mirror pypowsybl.dynamic's expected DataFrame schemas)
# ---------------------------------------------------------------------------

DYNAMIC_MODEL_MAPPING_COLS = {"static_id", "dynamic_model_id", "parameter_set_id"}
EVENT_MAPPING_COLS = {"static_id", "event_model_id", "parameter_set_id"}
VARIABLE_MAPPING_COLS = {"dynamic_model_id", "variable"}


@dataclass
class DynawoMappings:
    """Dynawo-ready simulation inputs derived from DynamicInputs.

    The three DataFrames map directly onto pypowsybl.dynamic's APIs:
    - dynamic_model_mapping → ModelMapping.add_all_dynamic_mappings()
    - event_mapping         → EventMapping / add_event()
    - variable_mapping      → CurveMapping / add_curve()

    Attributes
    ----------
    dynamic_model_mapping : pd.DataFrame
        Columns: static_id, dynamic_model_id, parameter_set_id.
        One row per network element to be equipped with a dynamic model.
    event_mapping : pd.DataFrame
        Columns: static_id, event_model_id, parameter_set_id.
        One row per event in the simulation sequence.
    variable_mapping : pd.DataFrame
        Columns: dynamic_model_id, variable.
        One row per monitored output variable (curve).
    """

    dynamic_model_mapping: pd.DataFrame
    event_mapping: pd.DataFrame
    variable_mapping: pd.DataFrame

    def validate(self) -> None:
        """Assert required columns are present with correct types.

        Raises
        ------
        ValueError
            With a descriptive message listing missing columns.
        """
        _check_cols(
            self.dynamic_model_mapping,
            DYNAMIC_MODEL_MAPPING_COLS,
            "dynamic_model_mapping",
        )
        _check_cols(self.event_mapping, EVENT_MAPPING_COLS, "event_mapping")
        _check_cols(self.variable_mapping, VARIABLE_MAPPING_COLS, "variable_mapping")


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def generate_dynawo_mappings(dynamic_inputs: DynamicInputs) -> DynawoMappings:
    """Convert generic DynamicInputs into Dynawo-compatible DynawoMappings.

    The conversion is a 1-to-1 column rename/passthrough because
    DynamicInputs already uses the same column schema as DynawoMappings.
    Element IDs in static_id are already pypowsybl IDs (as loaded from the
    CSV); no additional ID translation is needed here — ID resolution via
    p2g_maps happens upstream in _prepare_network_and_scenarios_dynamic.

    Parameters
    ----------
    dynamic_inputs : DynamicInputs
        Generic inputs loaded by load_raw_inputs().

    Returns
    -------
    DynawoMappings
        Validated Dynawo-ready mapping dataframes.
    """
    mappings = DynawoMappings(
        dynamic_model_mapping=dynamic_inputs.dynamic_models.copy(),
        event_mapping=dynamic_inputs.events.copy(),
        variable_mapping=dynamic_inputs.variables.copy(),
    )
    mappings.validate()
    return mappings


def prepare_dynawo_parameters(config) -> Any:
    """Build a pypowsybl.dynamic.Parameters object from config.

    Reads ``config.dynamic.solver_parameters`` and constructs the
    pypowsybl parameters object. All keys that do not match known
    Parameters fields are silently ignored so that the config can carry
    extra annotations without breaking.

    Parameters
    ----------
    config : NestedNamespace
        Must have a ``dynamic.solver_parameters`` attribute with keys:
        - solver_type : "SIM" (simplified, default) or "IDA"
        - start_time  : float, simulation start time in seconds
        - stop_time   : float, simulation stop time in seconds
        - precision   : float, solver precision (default 1e-6)

    Returns
    -------
    pypowsybl.dynamic.Parameters
    """
    check_pypowsybl_dynamic_available()
    dyn = _get_pypowsybl_dynamic()

    sp = getattr(config.dynamic, "solver_parameters", None) or {}
    # NestedNamespace → dict, or already a dict
    if hasattr(sp, "to_dict"):
        sp = sp.to_dict()

    start_time = float(sp.get("start_time", 0.0))
    stop_time = float(sp.get("stop_time", 10.0))
    precision = float(sp.get("precision", 1.0e-6))
    solver_type = str(sp.get("solver_type", "SIM")).upper()

    params = dyn.Parameters(
        start_time=start_time,
        stop_time=stop_time,
        precision=precision,
    )

    # Set solver type if the API exposes it as an attribute
    if hasattr(params, "solver_type"):
        params.solver_type = solver_type

    return params


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _check_cols(df: pd.DataFrame, required: set[str], label: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"DynawoMappings.{label} is missing required columns: {sorted(missing)}. "
            f"Present columns: {list(df.columns)}",
        )


__all__ = [
    "DynawoMappings",
    "generate_dynawo_mappings",
    "prepare_dynawo_parameters",
    "DYNAMIC_MODEL_MAPPING_COLS",
    "EVENT_MAPPING_COLS",
    "VARIABLE_MAPPING_COLS",
]
