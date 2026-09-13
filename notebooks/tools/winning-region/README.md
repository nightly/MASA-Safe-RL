# Winning-region explorer

A Jupyter-launched, offline SVG GUI for watching a deterministic safety shield's
greatest fixed point emerge, one product state at a time.

## Launch

From the repository root, in the Python environment used by your notebook kernel:

```sh
python -m pip install -e .
jupyter notebook notebooks/tools/winning-region/winning_region.ipynb
```

Run the notebook. It opens the GUI in your local browser. For a remote kernel,
set `OPEN_IN_BROWSER = False` to embed the GUI in the notebook instead. Re-run the
notebook's code cells after trusting a downloaded notebook. Browser restrictions
on local files can also be avoided with the embedded mode.

The default is a small, slippery FrozenLake and the safety property **G ¬hole**.
Change the environment or the bad-prefix DFA in the configuration cell. There is
no string-to-LTL compiler: use MASA's `DFA` and propositional guard objects to
express the safety property. Accepting DFA states mean **violations**, not goals.

No new runtime dependencies are needed beyond the project's NumPy, IPython and
Jupyter dependencies. Rendering uses native browser SVG, not a server, Graphviz,
JavaScript CDN, or a second desktop event loop. Source Sans Pro / Source Sans 3
are used when installed locally, with system-font fallbacks. Fonts are not
bundled or downloaded.

## Explore

- **Play, Back, Step, Restart, Next round, Fixed point**, a speed selector and a
  scrubber provide reversible playback. Space toggles playback, arrows step,
  Home rewinds and End jumps to convergence when a form control is not focused.
- **Click a node** or use the state selector to inspect each action and its full
  support. Hover an action to isolate its edges. Enable **Follow the computation**
  to follow the current state again. Click the background to clear selection.
- **Drag nodes**, pan the background, scroll to zoom, or use Fit view. Positions
  remain stable when stepping or scrubbing. All action labels can be toggled.
- **Export diagram SVG** saves the current product graph. **Export trace JSON**
  saves the model, events, elimination rounds and final shield action mask.

The safety monitor is shown separately. Its guards follow MASA's insertion-order
transition semantics; `otherwise` denotes an implicit self-loop. Product states
are arranged in monitor-state lanes. Environments with a two-dimensional `desc`
retain their row-major geometry; other environments use a deterministic grid.
Long edge/node labels are abbreviated visually and remain available in tooltips
and the inspector.

## Use another environment

With this directory on `sys.path`:

```python
from explorer import build_explorer

view = build_explorer(
    env,                         # raw, unwrapped finite MASA environment
    dfa,                         # MASA bad-prefix DFA
    property_name="My safety property",
    # label_fn=lambda s: {...},  # defaults to env.label_fn
    # action_names={0: "Wait", 1: "Advance"},
    # state_names={0: "Entrance", 1: "Bridge"},
)
view.open()                      # standalone local GUI; returns the HTML path
# view.show(height=900)          # isolated iframe for remote notebooks
# view.save("my-exploration.html")
winning = view.trace.winning
allowed_actions = view.trace.allowed
```

`action_names()` may return a sequence or an integer-keyed mapping. The resolver
also understands `get_action_meanings()`. Explicit names take priority; missing
names fall back to `Action N`, never to guessed directions. FrozenLake exposes
`("Left", "Down", "Right", "Up")` through `action_names()`.

The environment must expose zero-based scalar Discrete state/action spaces and
MASA's exact dense `P[next_state, state, action]` model or sparse successor model.
Pass the unwrapped model, without observation/action remapping. The builder never
calls `reset`, `step` or `close` on your environment, or advances your DFA.

## Computation and interpretation

The explorer reuses `masa.deterministic_shield.support.read_support` and the
shield's product indexing `q_index * n_states + s`. The monitor has already
consumed `L(s)`; a product transition consumes **L(s')**, matching the actual
shield rather than the older generic LTL product helpers. Reset product IDs are
shown in the inspector for each base state, not as a claim about the environment's
initial-state distribution.

Start from all non-rejecting product states. In each synchronous round, retain a
state if there exists an **enabled** action whose **every** supported successor
remains in the old candidate set. Amber removals are only proposals: the complete
round is committed together. This avoids accidentally animating a different,
in-place algorithm. Rejecting states, disabled actions, deadlocks, self-loops,
padding and all positive-probability outcomes are handled explicitly. The final
winning and allowed-action masks are checked against MASA's production solver on
every build; a mismatch raises an error instead of displaying a verified badge.

A candidate is not yet proved winning. A shield can allow multiple actions even
though its safety guarantee is deterministic. Probabilities affect support but
not the universal predecessor test. Terminal states need explicit model dynamics
(normally absorbing); time limits do not shorten the infinite safety horizon.
Safety does not require eventual progress to a goal.

The **full product** is synthesized, including unreachable combinations. Defaults
limit the GUI to 500 product states, 20,000 merged drawn edges and 200,000 trace
events. Limits raise helpful errors, never silently prune the computation. Raise
`max_states`, `max_edges` or `max_events` deliberately for larger examples; dense
graphs will become harder to read. The trace engine is independently available as
`compute_trace(targets, support, rejecting)` and uses no GUI imports.

## Tests

```sh
python -m pytest tests/tools/test_winning_region_explorer.py -q
```

The tests cover synchronous cascades, universal support, disabled actions,
padding, rejecting/empty winning sets, randomized agreement with the production
solver, adapter label timing, input/resource limits, action names, safe HTML
serialization, and an integration comparison with the actual FrozenLake shield.
The integration test is skipped when Gymnasium is not installed.
