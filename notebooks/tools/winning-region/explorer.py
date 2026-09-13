"""Notebook/browser explorer for MASA's deterministic safety winning region.

No GUI event loop, server, CDN, or optional dependency is required. The trace
engine can also be used without importing Gymnasium or MASA.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import html
import json
import math
from pathlib import Path
import tempfile
from typing import Any
import warnings
import webbrowser

import numpy as np


@dataclass(frozen=True)
class Trace:
    """A synchronous elimination trace, with compact, replayable events.

    ``removed_at`` is 0 for rejecting states, a positive elimination round for
    losing states, and -1 for fixed-point survivors. A round's decisions always
    read the *same* old winning set; removal is committed only at its end.
    """

    events: tuple[dict[str, Any], ...]
    removed_at: np.ndarray
    decision_at: np.ndarray
    winning: np.ndarray
    allowed: np.ndarray


def compute_trace(targets, support, rejecting, *, max_events: int = 200_000) -> Trace:
    """Trace ``nu W. safe ∩ {x | exists enabled a: Succ(x,a) ⊆ W}``.

    Array conventions match ``masa.deterministic_shield.winning_region``.
    Disabled actions are never safe. Padding is ignored. No probability cutoff
    is used: the caller supplies the complete positive-probability support.
    """
    targets = np.asarray(targets)
    support = np.asarray(support, dtype=bool)
    rejecting = np.asarray(rejecting, dtype=bool)
    if targets.ndim != 3 or targets.dtype.kind not in "iu":
        raise ValueError("targets must be an integer array of shape (DFA, state, slot).")
    nq, ns, width = targets.shape
    if min(nq, ns, width) < 1:
        raise ValueError("The model must contain states and at least one successor slot.")
    if (support.ndim != 3 or support.shape[0] != ns
            or support.shape[2] != width or support.shape[1] < 1):
        raise ValueError("support must have shape (state, action, slot), with actions > 0.")
    if rejecting.shape != (nq,):
        raise ValueError("rejecting must have one entry per DFA state.")
    if np.any(targets >= nq * ns) or np.any(targets < 0):
        raise ValueError("Every target, including padding, must be a valid product-state ID.")
    if max_events < 2:
        raise ValueError("max_events must be at least 2.")

    winning = np.repeat(~rejecting, ns)
    removed = np.where(winning, -1, 0)
    decision = np.full(nq * ns, -1, dtype=np.intp)
    allowed = np.zeros((nq * ns, support.shape[1]), dtype=bool)
    enabled = support.any(axis=2)
    events: list[dict[str, Any]] = []

    def emit(phase: str, round_: int, state: int | None = None) -> None:
        if len(events) >= max_events:
            raise ValueError("Trace exceeds max_events; use a smaller model or raise the limit.")
        events.append({"phase": phase, "round": round_, "state": state})

    emit("initial", 0)
    for round_ in range(1, nq * ns + 2):
        updated = winning.copy()
        allowed.fill(False)
        for state in np.flatnonzero(winning):
            state = int(state)
            q, s = divmod(state, ns)
            emit("inspect", round_, state)
            escapes = (support[s] & ~winning[targets[q, s]][None, :]).any(axis=1)
            allowed[state] = enabled[s] & ~escapes
            keep = bool(allowed[state].any())
            emit("keep" if keep else "remove", round_, state)
            updated[state] = keep
            if not keep:
                removed[state] = round_
                decision[state] = len(events) - 1
        if np.array_equal(updated, winning):
            emit("fixed", round_)
            break
        emit("commit", round_)
        winning = updated
    # Immutable results prevent an exported trace drifting after construction.
    for array in (removed, decision, winning, allowed):
        array.flags.writeable = False
    return Trace(tuple(events), removed, decision, winning, allowed)


def resolve_action_names(env, count: int, override=None) -> list[str]:
    """Use explicit names, then action_names(), then get_action_meanings().

    Names may be a sequence or an integer-keyed mapping. Partial mappings fall
    back to 'Action N'; no movement direction is ever guessed from an integer.
    Gymnasium wrappers are traversed without calling reset() or step().
    """
    value = override
    current, seen = env, set()
    while value is None and current is not None and id(current) not in seen:
        seen.add(id(current))
        for attribute in ("action_names", "get_action_meanings"):
            candidate = getattr(current, attribute, None)
            if candidate is not None:
                value = candidate() if callable(candidate) else candidate
                if value is not None:
                    break
        current = getattr(current, "env", None)
    if callable(value):
        value = value()
    if value is None:
        return [f"Action {a}" for a in range(count)]
    if isinstance(value, Mapping):
        return [str(value.get(a, f"Action {a}")) for a in range(count)]
    if isinstance(value, (str, bytes)) or not hasattr(value, "__len__") or len(value) != count:
        raise ValueError(f"action_names must be a mapping or a sequence of {count} names.")
    return [str(name) for name in value]


def _guard_text(guard) -> str:
    """Readable MASA propositional guards, without object-address repr strings."""
    kind = type(guard).__name__
    if kind == "Atom":
        return str(guard.atom)
    if kind == "Truth":
        return "true"
    if kind == "Neg":
        return f"¬({_guard_text(guard.subformula)})"
    operator = {"And": "∧", "Or": "∨", "Implies": "→"}.get(kind)
    if operator:
        return f"({_guard_text(guard.subformula_1)} {operator} {_guard_text(guard.subformula_2)})"
    return kind


def _layout(ns: int, nq: int, grid: tuple[int, int] | None) -> dict[str, Any]:
    """Stable DFA lanes; row-major grids retain the environment's geometry."""
    rows, cols = grid or (math.ceil(ns / math.ceil(math.sqrt(ns))), math.ceil(math.sqrt(ns)))
    lane_w, lane_h = max(380, cols * 138 + 100), max(300, rows * 126 + 130)
    lane_cols = min(nq, 2)
    nodes, lanes = [], []
    for q in range(nq):
        left, top = (q % lane_cols) * lane_w, (q // lane_cols) * lane_h
        lanes.append({"q": q, "x": left + 18, "y": top + 18,
                      "width": lane_w - 36, "height": lane_h - 36})
        for s in range(ns):
            nodes.append({"x": left + 78 + (s % cols) * 138,
                          "y": top + 106 + (s // cols) * 126})
    return {"nodes": nodes, "lanes": lanes, "width": lane_cols * lane_w,
            "height": math.ceil(nq / lane_cols) * lane_h}


class Explorer:
    """A self-contained visualization; no environment reference is retained."""

    def __init__(self, data: dict[str, Any], trace: Trace):
        self.data = data
        self.trace = trace

    def to_html(self) -> str:
        template = Path(__file__).with_name("viewer.html").read_text(encoding="utf-8")
        # JSON lives in a script element: escape HTML delimiters even in labels.
        payload = json.dumps(self.data, ensure_ascii=True, allow_nan=False)
        payload = payload.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
        return template.replace("__WINNING_REGION_DATA__", payload)

    def save(self, path: str | Path) -> Path:
        """Export an offline HTML document; an existing target file is replaced."""
        path = Path(path).expanduser().resolve()
        path.write_text(self.to_html(), encoding="utf-8")
        return path

    def open(self, path: str | Path | None = None) -> Path:
        """Open in the local browser, returning the HTML path for manual use."""
        if path is None:
            with tempfile.NamedTemporaryFile(prefix="winning-region-", suffix=".html", delete=False) as file:
                path = file.name
        result = self.save(path)
        if not webbrowser.open(result.as_uri(), new=2):
            warnings.warn(f"Browser did not open. Open {result} manually or use show().", stacklevel=2)
        return result

    def show(self, *, height: int = 900):
        """Embed in Jupyter, including kernels running on a remote machine."""
        from IPython.display import HTML, display
        if height < 200:
            raise ValueError("height must be at least 200 pixels.")
        # Isolate styles and JavaScript from the notebook. Scripts can only see
        # this frame; same-origin access and remote network resources are unused.
        frame = (f'<iframe title="Winning region explorer" sandbox="allow-scripts allow-downloads" '
                 f'style="width:100%;height:{int(height)}px;border:0;border-radius:16px" '
                 f'srcdoc="{html.escape(self.to_html(), quote=True)}"></iframe>')
        return display(HTML(frame))


def build_explorer(
    env,
    dfa,
    *,
    property_name: str = "Safety property",
    label_fn: Callable[[int], Any] | None = None,
    action_names: Sequence[str] | Mapping[int, str] | None = None,
    state_names: Sequence[str] | Mapping[int, str] | None = None,
    max_states: int = 500,
    max_edges: int = 20_000,
    max_events: int = 200_000,
) -> Explorer:
    """Build from a raw finite MASA environment and a bad-prefix MASA DFA.

    Accepting DFA states mean *violations*, not goals. Pass an unwrapped base
    environment with zero-based Discrete states/actions and an exact MASA
    transition model. The monitor consumes L(s) on reset, then L(s') on a step,
    matching the deterministic shield (not the older generic product helpers).
    Terminal states must have explicit model dynamics, usually self-loops.
    This function never resets/steps/closes the caller's environment or DFA.
    """
    from masa.deterministic_shield.support import read_support
    from masa.deterministic_shield.winning_region import winning_region

    base = getattr(env, "unwrapped", env)
    if base is not env:
        raise TypeError("Pass the unwrapped base environment, not an observation/action wrapper.")
    for space in (env.observation_space, env.action_space):
        if (not hasattr(space, "n") or getattr(space, "start", 0) != 0
                or getattr(space, "shape", ()) != ()):
            raise TypeError("States/actions must be zero-based scalar Discrete spaces.")
    ns, na = int(env.observation_space.n), int(env.action_space.n)
    states = list(dfa.states)
    nq = len(states)
    if not nq or len(set(states)) != nq or ns < 1 or na < 1:
        raise ValueError("The model and DFA need nonempty, unique states and actions.")
    if ns * nq > max_states:
        raise ValueError(f"{ns * nq} product states exceeds max_states={max_states}; "
                         "use a smaller model or deliberately raise the limit. No states are truncated.")
    index = {q: i for i, q in enumerate(states)}
    if dfa.initial not in index or not set(dfa.accepting).issubset(index):
        raise ValueError("Initial/accepting DFA states must belong to dfa.states.")
    if dfa.initial in dfa.accepting:
        raise ValueError("The DFA already rejects the empty prefix, as rejected by MASA's shield.")
    label_fn = label_fn or getattr(env, "label_fn", None)
    if not callable(label_fn):
        raise TypeError("Provide label_fn(state) or an environment with label_fn().")
    labels = [set(label_fn(s)) for s in range(ns)]
    next_q = np.asarray([[index[dfa.transition(q, label)] for label in labels]
                         for q in states], dtype=np.intp)
    successors, support = read_support(base, ns, na)
    targets = next_q[:, successors] * ns + successors
    rejecting = np.asarray([q in dfa.accepting for q in states], dtype=bool)
    trace = compute_trace(targets, support, rejecting, max_events=max_events)
    expected_w, expected_a = winning_region(targets, support, rejecting)
    if not (np.array_equal(trace.winning, expected_w) and np.array_equal(trace.allowed, expected_a)):
        raise RuntimeError("Visualization trace disagrees with MASA's production shield solver.")

    names = resolve_action_names(env, na, action_names)
    if state_names is None:
        state_names = [f"s{s}" for s in range(ns)]
    if isinstance(state_names, Mapping):
        state_names = [state_names.get(s, f"s{s}") for s in range(ns)]
    if isinstance(state_names, (str, bytes)) or len(state_names) != ns:
        raise ValueError(f"state_names must provide {ns} entries or an integer-keyed mapping.")
    # Every action retains its complete support; merging parallel drawn edges
    # is only presentation, never a change to the predecessor computation.
    adjacency, edges = [], {}
    for q in range(nq):
        for s in range(ns):
            source = q * ns + s
            actions = []
            for a in range(na):
                dests = sorted({int(t) for t in targets[q, s, support[s, a]]})
                actions.append(dests)
                for target in dests:
                    edges.setdefault((source, target), []).append(a)
                    if len(edges) > max_edges:
                        raise ValueError("Graph exceeds max_edges; use a smaller model or raise the limit.")
            adjacency.append(actions)
    monitor_edges = []
    for q in states:
        outgoing = dfa.edges[q]
        for order, (target, guard) in enumerate(outgoing.items(), 1):
            monitor_edges.append({"source": index[q], "target": index[target],
                                  "label": _guard_text(guard), "order": order})
        if q not in outgoing:
            monitor_edges.append({"source": index[q], "target": index[q],
                                  "label": "otherwise", "order": len(outgoing) + 1})
    desc = getattr(base, "desc", None)
    grid = tuple(np.asarray(desc).shape) if desc is not None else None
    if grid is not None and (len(grid) != 2 or math.prod(grid) != ns):
        grid = None
    initial = (next_q[index[dfa.initial]] * ns + np.arange(ns)).tolist()
    data = {
        "title": str(property_name), "environment": type(base).__name__,
        "nStates": ns, "nDfa": nq, "actions": names,
        "states": [str(name) for name in state_names], "dfa": [str(q) for q in states],
        "labels": [sorted(str(label) for label in labels_) for labels_ in labels],
        "rejecting": rejecting.tolist(), "initialDfa": index[dfa.initial],
        "initialProduct": initial, "adjacency": adjacency,
        "edges": [{"source": s, "target": t, "actions": a} for (s, t), a in edges.items()],
        "monitorEdges": monitor_edges, "layout": _layout(ns, nq, grid),
        "events": list(trace.events), "removedAt": trace.removed_at.tolist(),
        "decisionAt": trace.decision_at.tolist(), "winning": trace.winning.tolist(),
        "allowed": trace.allowed.tolist(), "verified": True,
    }
    return Explorer(data, trace)
