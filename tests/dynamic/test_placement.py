"""Tests for gridfm_datakit.dynamic.placement."""

from pathlib import Path

import numpy as np
import pytest

from markers import needs_powsybl

from gridfm_datakit.dynamic.placement import (
    MAX_ANCHOR_ATTEMPTS,
    EventGraph,
    PlacementError,
    place_scenario,
)

pytestmark = needs_powsybl

PATH_NETWORK_IEEE14 = str(
    Path(__file__).parent
    / "dynawo/benchmark_data/ieee14/ieee14_GeneratorDisconnections/IEEE14.iidm",
)

BUS2 = "_BUS____2_TN"
HOPS_FROM_BUS2 = {
    0: [2],
    1: [1, 3, 4, 5],
    2: [6, 7, 9],
    3: [8, 10, 11, 12, 13, 14],
}
TRANSFORMERS = {
    "_BUS____4-BUS____9-1_PT",
    "_BUS____4-BUS____7-1_PT",
    "_BUS____5-BUS____6-1_PT",
}
SEEDS = range(50)


def _bus(number):
    return f"_BUS__{number:_>3}_TN"


@pytest.fixture
def network():
    import pypowsybl.network as pn

    return pn.load(PATH_NETWORK_IEEE14)


@pytest.fixture
def graph(network):
    return EventGraph.from_network(network)


def _draw(graph, anchor, targets, seed):
    return place_scenario(graph, anchor, targets, np.random.default_rng(seed))


def test_distances_are_hop_counts(graph):
    expected = {
        _bus(number): hops
        for hops, numbers in HOPS_FROM_BUS2.items()
        for number in numbers
    }
    assert graph.distances(BUS2) == expected


def test_generators_one_hop_from_bus_2(graph):
    drawn = {_draw(graph, BUS2, [("generator", 1, 1)], s)[1][0] for s in SEEDS}
    assert drawn == {"_GEN____1_SM", "_GEN____3_SM"}


def test_lines_touching_bus_2(graph, network):
    lines = network.get_lines(
        attributes=["bus_breaker_bus1_id", "bus_breaker_bus2_id"],
    )
    touching = set(
        lines[
            (lines["bus_breaker_bus1_id"] == BUS2)
            | (lines["bus_breaker_bus2_id"] == BUS2)
        ].index,
    )
    drawn = {_draw(graph, BUS2, [("line", 0, 0)], s)[1][0] for s in SEEDS}
    assert drawn and drawn <= touching


def test_transformers_one_hop_from_bus_2(graph):
    drawn = {_draw(graph, BUS2, [("transformer", 1, 1)], s)[1][0] for s in SEEDS}
    assert drawn <= TRANSFORMERS and drawn


def test_targets_of_one_scenario_are_distinct(graph):
    for seed in SEEDS:
        _, (first, second) = _draw(
            graph,
            BUS2,
            [("generator", 0, 1), ("generator", 0, 1)],
            seed,
        )
        assert first != second


def test_random_anchor_is_reproducible_and_respects_distances(graph, network):
    targets = [("generator", 1, 2), ("load", 0, 1)]
    assert _draw(graph, None, targets, 7) == _draw(graph, None, targets, 7)

    generators = network.get_generators(attributes=["bus_breaker_bus_id"])
    loads = network.get_loads(attributes=["bus_breaker_bus_id"])
    bus_of = {
        **generators["bus_breaker_bus_id"].to_dict(),
        **loads["bus_breaker_bus_id"].to_dict(),
    }
    anchors = set()
    for seed in SEEDS:
        anchor, placed = _draw(graph, None, targets, seed)
        anchors.add(anchor)
        distance = graph.distances(anchor)
        for (_, low, high), element_id in zip(targets, placed):
            assert low <= distance[bus_of[element_id]] <= high
    assert len(anchors) > 1


def test_the_graph_sees_the_perturbed_variant(network):
    network.clone_variant("InitialState", "perturbed")
    network.set_working_variant("perturbed")
    network.update_branches(
        id=["_BUS____1-BUS____2-1_AC", "_BUS____1-BUS____5-1_AC"],
        connected1=[False, False],
        connected2=[False, False],
    )
    network.update_generators(id="_GEN____3_SM", connected=False)

    perturbed = EventGraph.from_network(network)
    assert _bus(1) not in perturbed.distances(BUS2)
    with pytest.raises(PlacementError, match=BUS2):
        _draw(perturbed, BUS2, [("generator", 1, 1)], 0)

    network.set_working_variant("InitialState")
    initial = EventGraph.from_network(network)
    assert _draw(initial, BUS2, [("generator", 1, 1)], 0)[0] == BUS2


def test_an_unplaceable_random_scenario_gives_up(graph):
    with pytest.raises(PlacementError, match=str(MAX_ANCHOR_ATTEMPTS)):
        _draw(graph, None, [("generator", 20, 20)], 0)


@pytest.mark.parametrize(
    "anchor, targets",
    [
        ("_BUS___99_TN", [("generator", 0, 1)]),
        (BUS2, [("switch", 0, 0)]),
        (BUS2, [("load", 2, 1)]),
        (BUS2, [("load", -1, 1)]),
    ],
)
def test_invalid_requests_raise_value_error(graph, anchor, targets):
    with pytest.raises(ValueError):
        _draw(graph, anchor, targets, 0)
