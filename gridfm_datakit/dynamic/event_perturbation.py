"""Random dynamic events: the ``dynamic.event_perturbation`` block.

Form:
    type: none | file | random          required when the block is present
    n_event_variants: M                 positive int, default 1, 1 unless random
    scenarios:                          random only, exactly one
      - name: str
        anchor: random | <bus ID>       default random
        start_time: <value spec>        inside the solver window
        events:                         exactly one
          - type: <event type>
            target: {element: <element>, distance: int | [min, max]}
            params: {<param>: <value spec>}   every param, none for Disconnect

Event types, allowed elements, params and floors (``_EVENTS``):
    Disconnect                 every ELEMENT_TYPES
    NodeFault                  bus                 fault_time > 0, r_pu >= 0, x_pu >= 0
    ActivePowerVariation       generator, load     delta_p any sign
    ReactivePowerVariation     generator, load     delta_q any sign
    ReferenceVoltageVariation  generator           delta_u any sign

``_EVENTS`` copies the keys of ``dynawo.utils.EVENT_PARAMS_MAPPING``: importing
it here is circular. A test keeps them in sync.

A NodeFault ends at ``start_time + fault_time``, which must not exceed the stop
time. The other types are instantaneous.

An absent block is ``type: file``, the ``events_file`` rows.

Seeding: one ``default_rng([seed, scenario_index, perturbation_index,
event_index])`` per event variant, so a draw does not depend on chunking or
process count. Per scenario: placement, start time, then each event's params in
table order. Disconnect draws no params.

The M event variants of a topology variant share its balanced state: it does not
depend on the events, so it is computed once. A Dynawo run writes its final state
into the variant it ran on, so each event variant runs on its own clone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from gridfm_datakit.dynamic.placement import (
    ELEMENT_TYPES,
    EventGraph,
    PlacementError,
    place_scenario,
)
from gridfm_datakit.utils.param_handler import NestedNamespace
from gridfm_datakit.utils.value_spec import (
    NON_NEGATIVE,
    POSITIVE,
    Floor,
    ValueSpec,
    parse_value_spec,
)

EVENT_COLUMNS = ["event_name", "static_id", "start_time", "params"]

_PATH = "dynamic.event_perturbation"
_TYPES = ("none", "file", "random")
_ANY_NUMBER = Floor(-math.inf, strict=False)
_DISCONNECT_PARAMS = "disconnect_only=;"
_EVENTS = {
    "Disconnect": (ELEMENT_TYPES, ()),
    "NodeFault": (
        ("bus",),
        (("fault_time", POSITIVE), ("r_pu", NON_NEGATIVE), ("x_pu", NON_NEGATIVE)),
    ),
    "ActivePowerVariation": (("generator", "load"), (("delta_p", _ANY_NUMBER),)),
    "ReactivePowerVariation": (("generator", "load"), (("delta_q", _ANY_NUMBER),)),
    "ReferenceVoltageVariation": (("generator",), (("delta_u", _ANY_NUMBER),)),
}


@dataclass(frozen=True)
class EventTarget:
    """The element an event applies to, by hop distance from the anchor.

    Args:
        element: One of ``ELEMENT_TYPES``.
        low: The smallest allowed distance.
        high: The largest allowed distance.
    """

    element: str
    low: int
    high: int


@dataclass(frozen=True)
class EventSpec:
    """One event of a scenario.

    Args:
        type: The Dynawo event name.
        target: Where the event applies.
        params: The ``(param, spec)`` pairs of the type, in ``_EVENTS`` order.
    """

    type: str
    target: EventTarget
    params: Tuple[Tuple[str, ValueSpec], ...] = ()


@dataclass(frozen=True)
class EventScenario:
    """A set of events drawn around one anchor bus.

    Args:
        name: The scenario name, used in error messages.
        anchor: A bus ID, or ``None`` to draw one.
        start_time: The event time spec.
        events: The events of the scenario.
    """

    name: str
    anchor: Optional[str]
    start_time: ValueSpec
    events: Tuple[EventSpec, ...]


@dataclass(frozen=True)
class EventPerturbation:
    """The parsed ``dynamic.event_perturbation`` block.

    Args:
        type: ``none``, ``file`` or ``random``.
        n_event_variants: The event variants per topology variant.
        scenarios: The scenarios drawn in ``random`` mode.
    """

    type: str = "file"
    n_event_variants: int = 1
    scenarios: Tuple[EventScenario, ...] = ()


def parse_event_perturbation(
    block: Any,
    start_time: Optional[float],
    stop_time: Optional[float],
) -> EventPerturbation:
    """Parse the ``dynamic.event_perturbation`` block.

    Args:
        block: The block as a mapping or a ``NestedNamespace``, or ``None``.
        start_time: The simulation start time, ``None`` when not configured.
        stop_time: The simulation stop time, ``None`` when not configured.

    Returns:
        EventPerturbation: The parsed block, ``EventPerturbation()`` when absent.

    Raises:
        ValueError: If the block is malformed. The message starts with the config
            path of the offending entry.
    """
    if block is None:
        return EventPerturbation()
    if isinstance(block, NestedNamespace):
        block = block.to_dict()
    _check_keys(block, _PATH, {"type"}, {"n_event_variants", "scenarios"})

    kind = block["type"]
    if kind not in _TYPES:
        raise ValueError(f"{_PATH}.type: must be one of {list(_TYPES)}, got {kind!r}")

    n_event_variants = block.get("n_event_variants", 1)
    if (
        isinstance(n_event_variants, bool)
        or not isinstance(n_event_variants, int)
        or n_event_variants < 1
    ):
        raise ValueError(
            f"{_PATH}.n_event_variants: must be a positive integer, "
            f"got {n_event_variants!r}",
        )

    if kind != "random":
        if n_event_variants != 1:
            raise ValueError(
                f"{_PATH}.n_event_variants: must be 1 for type {kind!r}, "
                f"got {n_event_variants}",
            )
        if "scenarios" in block:
            raise ValueError(
                f"{_PATH}.scenarios: only accepted with type 'random', "
                f"got type {kind!r}",
            )
        return EventPerturbation(kind, n_event_variants)

    if start_time is None or stop_time is None:
        raise ValueError(
            f"{_PATH}: type 'random' needs dynamic.solver_parameters.start_time "
            "and stop_time to check event times",
        )
    scenarios = block.get("scenarios")
    if not isinstance(scenarios, list) or len(scenarios) != 1:
        raise ValueError(
            f"{_PATH}.scenarios: must be a list of exactly one scenario, "
            f"got {scenarios!r}",
        )
    scenario = _parse_scenario(
        scenarios[0],
        f"{_PATH}.scenarios[0]",
        float(start_time),
        float(stop_time),
    )
    return EventPerturbation(kind, n_event_variants, (scenario,))


def check_event_perturbation(
    perturbation: EventPerturbation,
    network_path: str,
) -> None:
    """Check that every fixed anchor can place its scenario on the base network.

    Args:
        perturbation: The parsed block.
        network_path: The network file.

    Raises:
        ValueError: If a fixed anchor is not a bus of the network or cannot place
            its targets. The message starts with the anchor's config path.
    """
    fixed = [
        (index, scenario)
        for index, scenario in enumerate(perturbation.scenarios)
        if scenario.anchor is not None
    ]
    if not fixed:
        return
    from gridfm_datakit.powsybl import load_net

    graph = EventGraph.from_network(load_net(network_path).pp_net)
    for index, scenario in fixed:
        try:
            place_scenario(
                graph,
                scenario.anchor,
                _targets(scenario),
                np.random.default_rng(0),
            )
        except (ValueError, PlacementError) as error:
            raise ValueError(
                f"{_PATH}.scenarios[{index}].anchor: {error}",
            ) from error


def draw_events(
    perturbation: EventPerturbation,
    graph: EventGraph,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Draw the events of one event variant.

    Args:
        perturbation: The parsed block.
        graph: The graph of the topology variant.
        rng: The generator of the event variant.

    Returns:
        pd.DataFrame: One row per event, columns ``EVENT_COLUMNS``.

    Raises:
        PlacementError: If a scenario cannot be placed.
    """
    rows = []
    for scenario in perturbation.scenarios:
        _, placed = place_scenario(graph, scenario.anchor, _targets(scenario), rng)
        start = float(scenario.start_time.sample(rng))
        for event, static_id in zip(scenario.events, placed):
            rows.append((event.type, static_id, start, _draw_params(event, rng)))
    return pd.DataFrame(rows, columns=EVENT_COLUMNS).astype({"start_time": float})


def _draw_params(event: EventSpec, rng: np.random.Generator) -> str:
    if not event.params:
        return _DISCONNECT_PARAMS
    return ";".join(f"{key}={float(spec.sample(rng))!r}" for key, spec in event.params)


def _targets(scenario: EventScenario) -> list:
    return [
        (event.target.element, event.target.low, event.target.high)
        for event in scenario.events
    ]


def _parse_scenario(
    block: Any,
    path: str,
    start_time: float,
    stop_time: float,
) -> EventScenario:
    _check_keys(block, path, {"name", "start_time", "events"}, {"anchor"})

    name = block["name"]
    if not isinstance(name, str) or not name:
        raise ValueError(f"{path}.name: must be a non-empty string, got {name!r}")

    anchor = block.get("anchor", "random")
    if not isinstance(anchor, str) or not anchor:
        raise ValueError(
            f"{path}.anchor: must be 'random' or a bus ID, got {anchor!r}",
        )

    spec = parse_value_spec(block["start_time"], f"{path}.start_time", POSITIVE)
    low, high = spec.support()
    if low < start_time or high > stop_time:
        raise ValueError(
            f"{path}.start_time: values in [{low}, {high}] fall outside the "
            f"simulation window [{start_time}, {stop_time}]",
        )

    events = block["events"]
    if not isinstance(events, list) or len(events) != 1:
        raise ValueError(
            f"{path}.events: must be a list of exactly one event, got {events!r}",
        )
    event = _parse_event(events[0], f"{path}.events[0]")
    fault_time = dict(event.params).get("fault_time")
    if fault_time is not None and high + fault_time.support()[1] > stop_time:
        raise ValueError(
            f"{path}.events[0].params.fault_time: a fault starting by {high} and "
            f"lasting up to {fault_time.support()[1]} ends after the stop time "
            f"{stop_time}",
        )
    return EventScenario(
        name=name,
        anchor=None if anchor == "random" else anchor,
        start_time=spec,
        events=(event,),
    )


def _parse_event(block: Any, path: str) -> EventSpec:
    _check_keys(block, path, {"type", "target"}, {"params"})
    kind = block["type"]
    if kind not in _EVENTS:
        raise ValueError(f"{path}.type: must be one of {list(_EVENTS)}, got {kind!r}")
    target = _parse_target(block["target"], f"{path}.target")
    elements, floors = _EVENTS[kind]
    if target.element not in elements:
        raise ValueError(
            f"{path}.target.element: {kind} accepts {list(elements)}, "
            f"got {target.element!r}",
        )

    if not floors:
        if "params" in block:
            raise ValueError(f"{path}.params: {kind} takes no params")
        return EventSpec(kind, target)
    if "params" not in block:
        raise ValueError(f"{path}.params: required key is missing")
    params = block["params"]
    _check_keys(params, f"{path}.params", {key for key, _ in floors}, set())
    return EventSpec(
        kind,
        target,
        tuple(
            (key, parse_value_spec(params[key], f"{path}.params.{key}", floor))
            for key, floor in floors
        ),
    )


def _parse_target(block: Any, path: str) -> EventTarget:
    _check_keys(block, path, {"element", "distance"}, set())
    element = block["element"]
    if element not in ELEMENT_TYPES:
        raise ValueError(
            f"{path}.element: must be one of {list(ELEMENT_TYPES)}, got {element!r}",
        )

    distance = block["distance"]
    bounds = distance if isinstance(distance, list) else [distance, distance]
    if (
        len(bounds) != 2
        or any(isinstance(b, bool) or not isinstance(b, int) for b in bounds)
        or bounds[0] < 0
        or bounds[0] > bounds[1]
    ):
        raise ValueError(
            f"{path}.distance: must be a non-negative integer or an inclusive "
            f"[min, max] of them, got {distance!r}",
        )
    return EventTarget(element, bounds[0], bounds[1])


def _check_keys(block: Any, path: str, required: set, optional: set) -> None:
    if not isinstance(block, Mapping):
        raise ValueError(f"{path}: must be a mapping, got {block!r}")
    unknown = sorted(set(block) - required - optional)
    if unknown:
        raise ValueError(
            f"{path}.{unknown[0]}: unknown key, accepted {sorted(required | optional)}",
        )
    missing = sorted(required - set(block))
    if missing:
        raise ValueError(f"{path}.{missing[0]}: required key is missing")


__all__ = [
    "EVENT_COLUMNS",
    "EventTarget",
    "EventSpec",
    "EventScenario",
    "EventPerturbation",
    "parse_event_perturbation",
    "check_event_perturbation",
    "draw_events",
]
