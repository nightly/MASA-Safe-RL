"""Headless trace tests; only the environment integration test needs Gymnasium."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "notebooks/tools/winning-region"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


explorer = load(TOOL / "explorer.py", "winning_region_explorer_test_module")
# Avoid importing unrelated learning frameworks through masa.__init__.
solver = load(ROOT / "masa/deterministic_shield/winning_region.py", "winning_region_reference_test_module")


@pytest.mark.parametrize("seed", range(100))
def test_trace_matches_production_solver(seed):
    rng = np.random.default_rng(seed)
    nq, ns, na, width = 3, 7, 4, 5
    targets = rng.integers(nq * ns, size=(nq, ns, width))
    support = rng.random((ns, na, width)) < .35
    rejecting = rng.random(nq) < .3
    trace = explorer.compute_trace(targets, support, rejecting)
    winning, allowed = solver.winning_region(targets, support, rejecting)
    np.testing.assert_array_equal(trace.winning, winning)
    np.testing.assert_array_equal(trace.allowed, allowed)
    assert trace.events[-1]["phase"] == "fixed"
    assert not trace.winning.flags.writeable
    assert not trace.allowed.flags.writeable
    assert np.all(trace.removed_at[~winning] >= 0)
    for event in trace.events:
        if event["phase"] not in {"inspect", "keep", "remove"}:
            continue
        state, round_ = event["state"], event["round"]
        assert trace.removed_at[state] == -1 or trace.removed_at[state] >= round_


def test_synchronous_cascade_and_disabled_actions():
    # One monitor state. 0 -> 1 -> 2 -> 3(deadlock), and 4 -> 4.
    targets = np.array([[[1], [2], [3], [3], [4]]])
    support = np.ones((5, 2, 1), dtype=bool)
    support[:, 1] = False
    support[3] = False
    trace = explorer.compute_trace(targets, support, [False])
    assert trace.removed_at.tolist() == [4, 3, 2, 1, -1]
    assert trace.winning.tolist() == [False, False, False, False, True]
    assert trace.allowed[-1].tolist() == [True, False]
    assert [e["round"] for e in trace.events if e["phase"] == "commit"] == [1, 2, 3, 4]
    for state in range(4):
        assert trace.events[trace.decision_at[state]]["phase"] == "remove"


def test_universal_support_not_existential_and_padding_ignored():
    # q1 is rejecting. From q0, a0 has BOTH a safe and an unsafe outcome.
    targets = np.array([[[0, 1, 1]], [[1, 1, 1]]])
    support = np.array([[[True, True, False], [True, False, False]]])
    trace = explorer.compute_trace(targets, support, [False, True])
    assert trace.winning.tolist() == [True, False]
    assert trace.allowed.tolist() == [[False, True], [False, False]]


def test_all_rejecting_and_immediate_fixed_point():
    targets, support = np.array([[[0]]]), np.array([[[True]]])
    assert explorer.compute_trace(targets, support, [True]).winning.tolist() == [False]
    trace = explorer.compute_trace(targets, support, [False])
    assert trace.removed_at.tolist() == [-1]
    assert [e["phase"] for e in trace.events] == ["initial", "inspect", "keep", "fixed"]


@pytest.mark.parametrize("targets,support,rejecting", [
    (np.ones((1, 1)), np.ones((1, 1, 1)), [False]),
    (np.ones((1, 1, 1)), np.ones((1, 1, 1)), [False]),
    (np.array([[[2]]]), np.ones((1, 1, 1)), [False]),
    (np.array([[[-1]]]), np.ones((1, 1, 1)), [False]),
    (np.array([[[0]]]), np.ones((2, 1, 1)), [False]),
    (np.array([[[0]]]), np.ones((1, 1, 1)), [False, True]),
    (np.array([[[0]]]), np.ones((1, 0, 1)), [False]),
])
def test_malformed_models_fail(targets, support, rejecting):
    with pytest.raises(ValueError):
        explorer.compute_trace(targets, support, rejecting)


def test_trace_budget_is_not_silent_truncation():
    with pytest.raises(ValueError, match="max_events"):
        explorer.compute_trace(np.array([[[0]]]), np.ones((1, 1, 1)), [False], max_events=2)


def test_names_use_metadata_and_never_guess():
    named = SimpleNamespace(action_names=lambda: {0: "Left", 2: "Wait"})
    wrapper = SimpleNamespace(env=named)
    assert explorer.resolve_action_names(wrapper, 3) == ["Left", "Action 1", "Wait"]
    assert explorer.resolve_action_names(object(), 2) == ["Action 0", "Action 1"]
    assert explorer.resolve_action_names(named, 2, ["Go", "Stop"]) == ["Go", "Stop"]
    with pytest.raises(ValueError):
        explorer.resolve_action_names(object(), 2, "left")
    with pytest.raises(ValueError):
        explorer.resolve_action_names(object(), 2, ["left"])


def test_html_labels_cannot_end_the_json_script(tmp_path):
    trace = explorer.compute_trace(np.array([[[0]]]), np.ones((1, 1, 1)), [False])
    text = '</script><script>alert("unsafe")</script>&'
    result = explorer.Explorer({"title": text}, trace)
    page = result.save(tmp_path / "explorer.html").read_text()
    assert text not in page
    payload = page.split('<script id="model" type="application/json">')[1].split('</script>')[0]
    assert json.loads(payload)["title"] == text
    assert "https://" not in page


def test_layout_is_stable_and_preserves_grid_geometry():
    first = explorer._layout(9, 2, (3, 3))
    assert first == explorer._layout(9, 2, (3, 3))
    assert len(first["nodes"]) == 18
    assert first["nodes"][0]["y"] == first["nodes"][1]["y"]
    assert first["nodes"][0]["x"] == first["nodes"][3]["x"]


def test_frozen_lake_adapter_matches_actual_shield():
    pytest.importorskip("gymnasium")
    from masa.common.constraints.ltl_safety import LTLSafetyEnv
    from masa.common.labelled_env import LabelledEnv
    from masa.common.ltl import Atom, DFA
    from masa.deterministic_shield.preemptive import PreemptiveLTLShield
    from masa.envs.tabular.frozen_lake import FrozenLake

    env = FrozenLake(desc=["SFF", "FHF", "FFG"], is_slippery=True)
    dfa = DFA(states=[0, 1], initial=0, accepting=[1])
    dfa.add_edge(0, 1, Atom("hole"))
    view = explorer.build_explorer(env, dfa, property_name="G ¬hole")
    # Synthesis itself must not initialize an episode or mutate the monitor.
    assert dfa.state == 0
    assert view.data["actions"] == ["Left", "Down", "Right", "Up"]
    # The numerical comparison in build_explorer is unconditional. A direct
    # wrapper comparison below also catches mismatched product-label timing.
    labelled = LabelledEnv(env, label_fn=env.label_fn)
    shield = PreemptiveLTLShield(LTLSafetyEnv(labelled, dfa=dfa, obs_type="discrete"))
    try:
        np.testing.assert_array_equal(view.trace.winning, shield.winning_region)
        np.testing.assert_array_equal(view.trace.allowed, shield.safe_actions)
    finally:
        shield.close()


@pytest.fixture
def finite_model(monkeypatch):
    """Exercise the adapter independently of GUI and environment dependencies."""
    class Atom:
        def __init__(self, atom):
            self.atom = atom

    class Model:
        observation_space = SimpleNamespace(n=5, start=0, shape=())
        action_space = SimpleNamespace(n=2, start=0, shape=())
        desc = None

        @property
        def unwrapped(self):
            return self

        def label_fn(self, s):
            return {"hazard"} if s == 3 else {"clear"}

        def action_names(self):
            return ["Advance", "Hold"]

        def reset(self, **kwargs):
            raise AssertionError("The explorer must not reset the environment.")

        def step(self, action):
            raise AssertionError("The explorer must not step the environment.")

    class Monitor:
        states = [10, 20]
        initial = 10
        accepting = [20]
        state = 10
        edges = {10: {20: Atom("hazard")}, 20: {}}

        def transition(self, q, labels):
            return 20 if q == 20 or "hazard" in labels else 10

    successors = np.array([[1], [2], [3], [3], [4]])
    support = np.ones((5, 2, 1), dtype=bool)
    support[:4, 1] = False
    monkeypatch.setitem(sys.modules, "masa.deterministic_shield.support",
                        SimpleNamespace(read_support=lambda base, ns, na: (successors, support)))
    monkeypatch.setitem(sys.modules, "masa.deterministic_shield.winning_region", solver)
    return Model(), Monitor()


def test_adapter_label_timing_initial_mapping_and_metadata(finite_model):
    env, dfa = finite_model
    view = explorer.build_explorer(env, dfa, property_name="G ¬hazard")
    assert view.data["initialProduct"] == [0, 1, 2, 8, 4]
    assert view.data["adjacency"][2][0] == [8]  # consumes the NEXT state's label
    assert view.trace.removed_at.tolist() == [3, 2, 1, 1, -1, 0, 0, 0, 0, 0]
    assert view.data["actions"] == ["Advance", "Hold"]
    assert view.data["dfa"] == ["10", "20"]
    assert dfa.state == 10
    assert view.data["verified"] is True


def test_adapter_limits_and_wrappers_are_explicit(finite_model):
    env, dfa = finite_model
    with pytest.raises(ValueError, match="max_states"):
        explorer.build_explorer(env, dfa, max_states=5)
    with pytest.raises(ValueError, match="max_edges"):
        explorer.build_explorer(env, dfa, max_edges=1)
    with pytest.raises(TypeError, match="unwrapped"):
        explorer.build_explorer(SimpleNamespace(unwrapped=env), dfa)
    with pytest.raises(ValueError, match="state_names"):
        explorer.build_explorer(env, dfa, state_names=["one"])


def test_adapter_does_not_advertise_an_unverified_trace(finite_model, monkeypatch):
    env, dfa = finite_model
    monkeypatch.setitem(sys.modules, "masa.deterministic_shield.winning_region",
                        SimpleNamespace(winning_region=lambda *args: (np.zeros(10), np.zeros((10, 2)))))
    with pytest.raises(RuntimeError, match="disagrees"):
        explorer.build_explorer(env, dfa)
