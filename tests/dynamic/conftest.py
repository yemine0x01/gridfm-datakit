"""
Shared fixtures for dynamic module tests.
"""

from __future__ import annotations

import pandas as pd
import pytest

import gridfm_datakit.powsybl as powsybl
from gridfm_datakit.dynamic import DynamicInputs
from gridfm_datakit.network import get_pglib_file_path, load_net_from_pglib

# ---------------------------------------------------------------------------
# Optional-dependency marks
# ---------------------------------------------------------------------------

requires_powsybl = pytest.mark.skipif(
    not powsybl.is_powsybl_available(),
    reason="pypowsybl is not installed. Install with: pip install gridfm-datakit[powsybl]",
)


def _is_pypowsybl_dynamic_available() -> bool:
    try:
        from gridfm_datakit.dynamic.dynawo.api import is_pypowsybl_dynamic_available

        return is_pypowsybl_dynamic_available()
    except Exception:
        return False


requires_pypowsybl_dynamic = pytest.mark.skipif(
    not _is_pypowsybl_dynamic_available(),
    reason=(
        "pypowsybl.dynamic is not installed. "
        "Install with: pip install 'pypowsybl[dynamic]'"
    ),
)


# ---------------------------------------------------------------------------
# Network fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gfm_net_ieee14():
    """gridfm_datakit Network — IEEE 14-bus."""
    return load_net_from_pglib("case14_ieee")


@pytest.fixture(scope="module")
def loaded_ieee14():
    """LoadedNetwork (pypowsybl + gridfm) — IEEE 14-bus."""
    if not powsybl.is_powsybl_available():
        pytest.skip("pypowsybl not installed")
    path = get_pglib_file_path("case14_ieee")
    return powsybl.load_net(path)


@pytest.fixture(scope="module")
def pp_net_ieee14(loaded_ieee14):
    """pypowsybl network for IEEE 14-bus."""
    return loaded_ieee14.pp_net


@pytest.fixture(scope="module")
def mapping_p2g_ieee14(loaded_ieee14):
    """MappingP2G for IEEE 14-bus."""
    return loaded_ieee14.mapping_p2g


# ---------------------------------------------------------------------------
# DynamicInputs / DynawoMappings fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sample_dynamic_inputs(loaded_ieee14):
    """Minimal DynamicInputs compatible with the IEEE 14-bus network.

    Uses pypowsybl generator IDs as static_id values.
    """
    if not powsybl.is_powsybl_available():
        pytest.skip("pypowsybl not installed")

    pp_net = loaded_ieee14.pp_net
    gen_ids = list(pp_net.get_generators().index)[:2]  # take first 2 generators
    line_ids = list(pp_net.get_lines().index)[:1]  # take first line

    dynamic_models = pd.DataFrame(
        {
            "static_id": gen_ids,
            "dynamic_model_id": ["GeneratorSynchronousThreeWindings"] * len(gen_ids),
            "parameter_set_id": [f"gen_{i}" for i in range(len(gen_ids))],
        },
    )

    events = pd.DataFrame(
        {
            "static_id": line_ids,
            "event_model_id": ["EventQuadripoleDisconnection"],
            "parameter_set_id": ["evt_0"],
        },
    )

    variables = pd.DataFrame(
        {
            "dynamic_model_id": gen_ids[:1],
            "variable": ["generator_omegaPu"],
        },
    )

    return DynamicInputs(
        dynamic_models=dynamic_models,
        events=events,
        variables=variables,
    )


@pytest.fixture(scope="module")
def sample_dynawo_mappings(sample_dynamic_inputs):
    """DynawoMappings derived from sample_dynamic_inputs."""
    from gridfm_datakit.dynamic.dynawo import generate_dynawo_mappings

    return generate_dynawo_mappings(sample_dynamic_inputs)
