"""
Dynamic simulation module for gridfm_datakit.

Provides a solver-agnostic layer for dynamic power system simulation data
generation. The first (and currently only) concrete backend is Dynaωo,
accessed through pypowsybl.dynamic.

Only the generic contracts (DynamicInputs, DynamicResults) and the
load_raw_inputs loader live here. Solver-specific logic is in submodules
(e.g. gridfm_datakit/dynamic/dynawo/).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Column schemas for DynamicInputs DataFrames
# ---------------------------------------------------------------------------
# These are the minimum required columns validated by load_raw_inputs when
# dynamic_solver == "dynawo". They map directly onto pypowsybl.dynamic's
# model-mapping / event / curve APIs.

DYNAMIC_MODELS_REQUIRED_COLS = {
    "static_id": str,  # pypowsybl element ID to equip with a dynamic model
    "dynamic_model_id": str,  # Dynawo model name  (e.g. "GeneratorSynchronousThreeWindings")
    "parameter_set_id": str,  # parameter set identifier (references a .par file group)
}

EVENTS_REQUIRED_COLS = {
    "static_id": str,  # pypowsybl element ID to which the event applies
    "event_model_id": str,  # Dynawo event model (e.g. "EventQuadripoleDisconnection")
    "parameter_set_id": str,  # parameter set identifier for event timing/severity
}

VARIABLES_REQUIRED_COLS = {
    "dynamic_model_id": str,  # dynamic model whose variable is monitored
    "variable": str,  # variable name to export as a curve (e.g. "generator_omegaPu")
}


# ---------------------------------------------------------------------------
# Generic data contracts
# ---------------------------------------------------------------------------


@dataclass
class DynamicInputs:
    """Solver-agnostic container for dynamic simulation inputs.

    All three attributes are pandas DataFrames so they remain compatible with
    pypowsybl.dynamic's native input format and are easy to inspect or
    serialize for debugging.

    Attributes
    ----------
    dynamic_models : pd.DataFrame
        One row per network element to be equipped with a dynamic model.
        Required columns: static_id, dynamic_model_id, parameter_set_id.
    events : pd.DataFrame
        One row per event in the simulation sequence.
        Required columns: static_id, event_model_id, parameter_set_id.
    variables : pd.DataFrame
        One row per monitored output variable (curve).
        Required columns: dynamic_model_id, variable.
    """

    dynamic_models: pd.DataFrame
    events: pd.DataFrame
    variables: pd.DataFrame


@dataclass
class DynamicResults:
    """Solver-agnostic container for dynamic simulation outputs.

    Attributes
    ----------
    dynamic_results : zarr.Array or zarr.Group
        Time-series output shaped (n_variables, n_timesteps) per scenario.
        Stored as an in-memory Zarr array during per-scenario processing;
        written to a persistent Zarr store by _save_generated_data.
    report : str
        Solver convergence / status text returned by the backend.
    """

    dynamic_results: Any  # zarr.Array or zarr.Group — (n_variables, n_timesteps)
    report: str


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_raw_inputs(config) -> DynamicInputs:
    """Load dynamic simulation inputs from CSV files declared in the config.

    Reads the three CSV (or Parquet) files listed under config.dynamic and
    returns a DynamicInputs instance. When dynamic_solver == "dynawo", the
    minimum required columns for each DataFrame are validated.

    Parameters
    ----------
    config : NestedNamespace
        Configuration object. Must have a ``dynamic`` attribute with:
        - dynamic_models_file : path to the models CSV
        - events_file         : path to the events CSV
        - variables_file      : path to the variables CSV
        - dynamic_solver      : solver name ("dynawo" or future alternatives)

    Returns
    -------
    DynamicInputs

    Raises
    ------
    FileNotFoundError
        If any of the three input files is missing.
    ValueError
        If required columns are absent from a DataFrame (Dynawo solver only).
    """
    dyn_cfg = config.dynamic

    dynamic_models = _load_table(dyn_cfg.dynamic_models_file)
    events = _load_table(dyn_cfg.events_file)
    variables = _load_table(dyn_cfg.variables_file)

    solver = getattr(dyn_cfg, "dynamic_solver", "dynawo")
    if solver == "dynawo":
        _validate_columns(
            dynamic_models,
            DYNAMIC_MODELS_REQUIRED_COLS,
            "dynamic_models_file",
        )
        _validate_columns(events, EVENTS_REQUIRED_COLS, "events_file")
        _validate_columns(variables, VARIABLES_REQUIRED_COLS, "variables_file")

    return DynamicInputs(
        dynamic_models=dynamic_models,
        events=events,
        variables=variables,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _load_table(path: str) -> pd.DataFrame:
    """Load a CSV or Parquet file into a DataFrame."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Dynamic input file not found: {path}")
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p)


def _validate_columns(
    df: pd.DataFrame,
    required: dict[str, type],
    file_label: str,
) -> None:
    """Assert that all required columns are present in df."""
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(
            f"{file_label} is missing required columns: {sorted(missing)}. "
            f"Present columns: {list(df.columns)}",
        )


__all__ = [
    "DynamicInputs",
    "DynamicResults",
    "load_raw_inputs",
    "DYNAMIC_MODELS_REQUIRED_COLS",
    "EVENTS_REQUIRED_COLS",
    "VARIABLES_REQUIRED_COLS",
]
