"""
Tests for the output format produced by the dynamic pipeline.

Covers:
- Zarr array shape (n_scenarios, n_variables, n_timesteps)
- Zarr concurrent write safety (no corruption)
- Parquet column schema matches static pipeline's BUS/BRANCH/GEN_COLUMNS
"""

from __future__ import annotations

import multiprocessing

import numpy as np
import pandas as pd
import pytest

from gridfm_datakit.dynamic import DynamicResults
from gridfm_datakit.utils.column_names import BRANCH_COLUMNS, BUS_COLUMNS, GEN_COLUMNS

zarr = pytest.importorskip(
    "zarr",
    reason="zarr is not installed. Install with: pip install gridfm-datakit[dynamic]",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dynamic_results(n_vars: int, n_t: int, report: str = "OK") -> DynamicResults:
    store = zarr.MemoryStore()
    root = zarr.open_group(store, mode="w")
    arr = np.random.rand(n_vars, n_t)
    root.create_dataset("curves", data=arr, dtype="float64")
    return DynamicResults(dynamic_results=root["curves"], report=report)


def _make_pf_data(
    scenario_index: int,
    n_buses: int = 14,
    n_branches: int = 20,
    n_gens: int = 5,
):
    """Return a minimal pf_data dict with correct column counts."""
    bus = np.zeros((n_buses, len(BUS_COLUMNS)))
    bus[:, 0] = scenario_index
    gen = np.zeros((n_gens, len(GEN_COLUMNS)))
    branch = np.zeros((n_branches, len(BRANCH_COLUMNS)))
    return {"bus": bus, "gen": gen, "branch": branch, "Y_bus": np.zeros((0, 4))}


# ---------------------------------------------------------------------------
# Zarr shape tests
# ---------------------------------------------------------------------------


class TestZarrShape:
    def test_zarr_shape_n_scenarios_n_vars_n_timesteps(self, tmp_path):
        """Zarr store must be shaped (n_scenarios, n_variables, n_timesteps)."""
        N_SCENARIOS, N_VARS, N_T = 5, 3, 100
        results = [
            {
                "scenario_index": i,
                "pf_data": _make_pf_data(i),
                "dynamic_results": _make_dynamic_results(N_VARS, N_T),
            }
            for i in range(N_SCENARIOS)
        ]

        # Use _save_generated_data directly
        from gridfm_datakit.dynamic.generate_dynamic import _save_generated_data
        from gridfm_datakit.utils.param_handler import NestedNamespace

        config = NestedNamespace(load=NestedNamespace(scenarios=N_SCENARIOS))
        config.to_dict = lambda: {}

        _save_generated_data(results, tmp_path, {}, config, seed=42)

        store = zarr.open(str(tmp_path / "dynamic_results.zarr"), mode="r")
        z = store["curves"]
        assert z.shape == (N_SCENARIOS, N_VARS, N_T), (
            f"Expected ({N_SCENARIOS}, {N_VARS}, {N_T}), got {z.shape}"
        )

    def test_zarr_dtype_is_float64(self, tmp_path):
        results = [
            {
                "scenario_index": 0,
                "pf_data": _make_pf_data(0),
                "dynamic_results": _make_dynamic_results(2, 50),
            },
        ]
        from gridfm_datakit.dynamic.generate_dynamic import _save_generated_data
        from gridfm_datakit.utils.param_handler import NestedNamespace

        config = NestedNamespace(load=NestedNamespace(scenarios=1))
        config.to_dict = lambda: {}
        _save_generated_data(results, tmp_path, {}, config, seed=0)
        store = zarr.open(str(tmp_path / "dynamic_results.zarr"), mode="r")
        assert store["curves"].dtype == np.float64


def _zarr_write_slice(args):
    """Module-level worker function for concurrent Zarr write test (must be picklable)."""
    idx, zarr_path, n_vars, n_t = args
    import numpy as np
    import zarr

    store = zarr.open(zarr_path, mode="r+")
    store["curves"][idx] = np.full((n_vars, n_t), float(idx))


class TestZarrConcurrentWrite:
    def test_no_corruption_with_multiple_workers(self, tmp_path):
        """Writing non-overlapping Zarr slices from multiple processes must not corrupt the store."""
        N_SCENARIOS, N_VARS, N_T = 8, 2, 50
        zarr_path = str(tmp_path / "concurrent.zarr")
        store = zarr.open(zarr_path, mode="w")
        store.create_dataset(
            "curves",
            shape=(N_SCENARIOS, N_VARS, N_T),
            dtype="float64",
            chunks=(1, N_VARS, N_T),
        )

        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=min(4, N_SCENARIOS)) as pool:
            pool.map(
                _zarr_write_slice,
                [(i, zarr_path, N_VARS, N_T) for i in range(N_SCENARIOS)],
            )

        store = zarr.open(zarr_path, mode="r")
        z = store["curves"]
        for i in range(N_SCENARIOS):
            expected = float(i)
            np.testing.assert_allclose(
                z[i],
                np.full((N_VARS, N_T), expected),
                err_msg=f"Scenario {i} corrupted in Zarr store",
            )


# ---------------------------------------------------------------------------
# Parquet schema tests
# ---------------------------------------------------------------------------


class TestParquetColumnsMatchStaticSchema:
    def test_bus_parquet_columns(self, tmp_path):
        """bus_data.parquet must have exactly the columns from BUS_COLUMNS (prefix match)."""
        from gridfm_datakit.dynamic.generate_dynamic import _save_generated_data
        from gridfm_datakit.utils.param_handler import NestedNamespace

        results = [
            {
                "scenario_index": 0,
                "pf_data": _make_pf_data(0),
                "dynamic_results": _make_dynamic_results(2, 10),
            },
        ]
        config = NestedNamespace(load=NestedNamespace(scenarios=1))
        config.to_dict = lambda: {}
        file_paths = {}
        _save_generated_data(results, tmp_path, file_paths, config, seed=0)

        df = pd.read_parquet(tmp_path / "bus_data.parquet")
        for col in BUS_COLUMNS[: df.shape[1]]:
            assert col in df.columns, f"Column {col!r} missing from bus_data.parquet"

    def test_branch_parquet_columns(self, tmp_path):
        from gridfm_datakit.dynamic.generate_dynamic import _save_generated_data
        from gridfm_datakit.utils.param_handler import NestedNamespace

        results = [
            {
                "scenario_index": 0,
                "pf_data": _make_pf_data(0),
                "dynamic_results": _make_dynamic_results(2, 10),
            },
        ]
        config = NestedNamespace(load=NestedNamespace(scenarios=1))
        config.to_dict = lambda: {}
        file_paths = {}
        _save_generated_data(results, tmp_path, file_paths, config, seed=0)

        df = pd.read_parquet(tmp_path / "branch_data.parquet")
        for col in BRANCH_COLUMNS[: df.shape[1]]:
            assert col in df.columns, f"Column {col!r} missing from branch_data.parquet"
