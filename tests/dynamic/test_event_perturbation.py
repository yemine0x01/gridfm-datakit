"""Tests for gridfm_datakit.dynamic.event_perturbation."""

import copy
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from markers import needs_powsybl

from gridfm_datakit.dynamic.event_perturbation import (
    EVENT_COLUMNS,
    EventPerturbation,
    EventTarget,
    check_event_perturbation,
    draw_events,
    parse_event_perturbation,
)
from gridfm_datakit.dynamic.placement import EventGraph
from gridfm_datakit.utils.param_handler import NestedNamespace

PATH = "dynamic.event_perturbation"
SCENARIO = f"{PATH}.scenarios[0]"
PATH_NETWORK_IEEE14 = str(
    Path(__file__).parent
    / "dynawo/benchmark_data/ieee14/ieee14_GeneratorDisconnections/IEEE14.iidm",
)


def _random(**overrides):
    block = {
        "type": "random",
        "n_event_variants": 3,
        "scenarios": [
            {
                "name": "generator_trip",
                "anchor": "random",
                "start_time": {"distribution": "uniform", "low": 20, "high": 80},
                "events": [
                    {
                        "type": "Disconnect",
                        "target": {"element": "generator", "distance": [0, 2]},
                    },
                ],
            },
        ],
    }
    block.update(overrides)
    return block


def _parse(block, start_time=0.0, stop_time=100.0):
    return parse_event_perturbation(block, start_time, stop_time)


def _raises(block, path, start_time=0.0, stop_time=100.0):
    with pytest.raises(ValueError, match=f"^{re.escape(path)}"):
        _parse(block, start_time, stop_time)


def _edit(edit):
    block = _random()
    edit(block)
    return block


class TestParse:
    def test_absent_block_is_file(self):
        assert _parse(None) == EventPerturbation()
        assert EventPerturbation().type == "file"

    @pytest.mark.parametrize("kind", ["none", "file"])
    def test_none_and_file(self, kind):
        assert _parse({"type": kind}) == EventPerturbation(kind, 1, ())

    def test_random(self):
        perturbation = _parse(_random())
        assert perturbation.type == "random"
        assert perturbation.n_event_variants == 3
        (scenario,) = perturbation.scenarios
        assert scenario.name == "generator_trip"
        assert scenario.anchor is None
        assert scenario.start_time.support() == (20.0, 80.0)
        (event,) = scenario.events
        assert event.type == "Disconnect"
        assert event.target == EventTarget("generator", 0, 2)

    def test_namespace_block_is_not_mutated(self):
        block = NestedNamespace(**_random())
        before = copy.deepcopy(block.to_dict())
        _parse(block)
        assert block.to_dict() == before

    def test_fixed_anchor_and_single_distance(self):
        def edit(block):
            block["scenarios"][0]["anchor"] = "_BUS____2_TN"
            block["scenarios"][0]["events"][0]["target"]["distance"] = 1

        scenario = _parse(_edit(edit)).scenarios[0]
        assert scenario.anchor == "_BUS____2_TN"
        assert scenario.events[0].target == EventTarget("generator", 1, 1)


class TestReject:
    @pytest.mark.parametrize("kind", ["none", "file"])
    def test_event_variants_outside_random(self, kind):
        _raises({"type": kind, "n_event_variants": 2}, f"{PATH}.n_event_variants")

    @pytest.mark.parametrize("value", [0, True, -1, 1.5])
    def test_invalid_event_variant_count(self, value):
        _raises(_random(n_event_variants=value), f"{PATH}.n_event_variants")

    def test_missing_type(self):
        _raises({"n_event_variants": 1}, f"{PATH}.type")

    def test_unknown_type(self):
        _raises({"type": "csv"}, f"{PATH}.type")

    def test_scenarios_outside_random(self):
        _raises({"type": "file", "scenarios": []}, f"{PATH}.scenarios")

    def test_random_needs_the_solver_window(self):
        _raises(_random(), PATH, start_time=None, stop_time=100.0)
        _raises(_random(), PATH, start_time=0.0, stop_time=None)

    @pytest.mark.parametrize(
        "edit, path",
        [
            (lambda b: b.update(seed=1), f"{PATH}.seed"),
            (lambda b: b["scenarios"][0].update(weight=1), f"{SCENARIO}.weight"),
            (
                lambda b: b["scenarios"][0]["events"][0].update(delay=1),
                f"{SCENARIO}.events[0].delay",
            ),
            (
                lambda b: b["scenarios"][0]["events"][0]["target"].update(
                    static_id="_GEN____1_SM",
                ),
                f"{SCENARIO}.events[0].target.static_id",
            ),
        ],
    )
    def test_unknown_keys_name_their_path(self, edit, path):
        with pytest.raises(ValueError, match=f"^{re.escape(path)}: unknown key"):
            _parse(_edit(edit))

    def test_a_second_scenario(self):
        _raises(
            _edit(lambda b: b["scenarios"].append(b["scenarios"][0])),
            f"{PATH}.scenarios",
        )

    def test_a_second_event(self):
        _raises(
            _edit(
                lambda b: b["scenarios"][0]["events"].append(
                    b["scenarios"][0]["events"][0],
                ),
            ),
            f"{SCENARIO}.events",
        )

    def test_another_event_type(self):
        _raises(
            _edit(lambda b: b["scenarios"][0]["events"][0].update(type="NodeFault")),
            f"{SCENARIO}.events[0].type",
        )

    def test_unknown_element(self):
        _raises(
            _edit(
                lambda b: b["scenarios"][0]["events"][0]["target"].update(
                    element="switch",
                ),
            ),
            f"{SCENARIO}.events[0].target.element",
        )

    @pytest.mark.parametrize("distance", [-1, [2, 1], True, [0, True], [1], "1"])
    def test_invalid_distance(self, distance):
        _raises(
            _edit(
                lambda b: b["scenarios"][0]["events"][0]["target"].update(
                    distance=distance,
                ),
            ),
            f"{SCENARIO}.events[0].target.distance",
        )

    def test_empty_name(self):
        _raises(_edit(lambda b: b["scenarios"][0].update(name="")), f"{SCENARIO}.name")

    @pytest.mark.parametrize(
        "spec",
        [{"distribution": "uniform", "low": 20, "high": 120}, -1],
    )
    def test_start_time_outside_the_window(self, spec):
        _raises(
            _edit(lambda b: b["scenarios"][0].update(start_time=spec)),
            f"{SCENARIO}.start_time",
        )


def _chain_graph():
    buses = [f"B{i}" for i in range(6)]
    adjacency = {
        bus: tuple(other for other in buses if abs(int(other[1:]) - int(bus[1:])) == 1)
        for bus in buses
    }
    return EventGraph(
        adjacency=adjacency,
        candidates={
            "bus": tuple((bus, (bus,)) for bus in buses),
            "generator": tuple((f"G{i}", (f"B{i}",)) for i in range(6)),
            "load": (),
            "line": tuple((f"L{i}", (f"B{i}", f"B{i + 1}")) for i in range(5)),
            "transformer": (),
        },
    )


class TestDrawEvents:
    def test_columns_and_values(self):
        graph = _chain_graph()
        perturbation = _parse(_random())
        frame = draw_events(perturbation, graph, np.random.default_rng([1, 0, 0, 0]))
        assert list(frame.columns) == EVENT_COLUMNS
        assert len(frame) == 1
        row = frame.iloc[0]
        assert row["event_name"] == "Disconnect"
        assert row["params"] == "disconnect_only=;"
        assert frame["start_time"].dtype == float

    def test_draws_are_reproducible_and_bounded(self):
        graph = _chain_graph()
        perturbation = _parse(_random())

        def draw(seed, event_index):
            rng = np.random.default_rng([seed, 0, 0, event_index])
            return draw_events(perturbation, graph, rng)

        differs = False
        for event_index in range(20):
            frame = draw(1, event_index)
            pd.testing.assert_frame_equal(frame, draw(1, event_index))
            differs |= not frame.equals(draw(2, event_index))
            start = frame["start_time"].iloc[0]
            assert 20 <= start < 80
            bus = f"B{frame['static_id'].iloc[0][1:]}"
            assert any(
                graph.distances(anchor).get(bus, 99) <= 2 for anchor in graph.adjacency
            )
        assert differs

    def test_no_scenario_draws_an_empty_frame(self):
        frame = draw_events(
            EventPerturbation(),
            _chain_graph(),
            np.random.default_rng(0),
        )
        assert frame.empty and list(frame.columns) == EVENT_COLUMNS


@needs_powsybl
class TestCheck:
    @staticmethod
    def _fixed(anchor, distance):
        def edit(block):
            block["scenarios"][0]["anchor"] = anchor
            block["scenarios"][0]["events"][0]["target"]["distance"] = distance

        return _parse(_edit(edit))

    def test_a_placeable_fixed_anchor_passes(self):
        check_event_perturbation(self._fixed("_BUS____2_TN", 1), PATH_NETWORK_IEEE14)

    @pytest.mark.parametrize(
        "anchor, distance",
        [("_BUS____2_TN", 20), ("_BUS___99_TN", 1)],
    )
    def test_an_unusable_fixed_anchor_raises(self, anchor, distance):
        with pytest.raises(ValueError, match=f"^{re.escape(SCENARIO)}.anchor: "):
            check_event_perturbation(
                self._fixed(anchor, distance),
                PATH_NETWORK_IEEE14,
            )

    def test_file_mode_does_not_load_the_network(self, tmp_path):
        check_event_perturbation(
            EventPerturbation("file"),
            str(tmp_path / "missing.iidm"),
        )
