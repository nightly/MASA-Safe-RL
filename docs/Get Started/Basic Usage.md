# Basic Usage

This page shows the minimal way to use MASA without `masa.common.utils.make_env`, by manually constructing a Gymnasium environment and wrapping it in the recommended order:

`gymnasium.wrappers.TimeLimit` -> `masa.common.labelled_env.LabelledEnv` -> `masa.common.constraints.base.BaseConstraintEnv` -> `masa.common.wrappers.ConstraintMonitor` -> `masa.common.wrappers.RewardMonitor`

This is the same order enforced by `masa.common.utils.make_env`, notably with `TimeLimit` first.

## Overview

MASA components reason over labels, which are atomic predicates derived from observations. The `masa.common.labelled_env.LabelledEnv` wrapper computes these labels on every `gymnasium.Env.reset` and `gymnasium.Env.step` and stores them in `info["labels"]`.

Constraints then consume these labels and expose consistent metrics, while monitor wrappers attach step and episode summaries to the `info` dictionary for logging and debugging.

## Minimal Environment Construction

```python
import gymnasium as gym

# Core MASA wrappers
from masa.common.wrappers import TimeLimit, ConstraintMonitor, RewardMonitor
from masa.common.labelled_env import LabelledEnv

# Simple Media Streaming environment
from masa.env.tabular.media_streaming import MediaStreaming

# Example constraint wrapper (a BaseConstraintEnv implementation)
from masa.common.constraints.cmdp import CumulativeCostEnv

# --- 1) Define label and cost functions ---

def label_fn(obs):
    """
    Example labelling function for MediaStreaming-like observations.

    Returns:
        set[str]: Atomic predicates holding in the current observation.
    """
    labels = set()
    # These keys are illustrative; adapt to your observation structure.
    try:
        if int(obs) == 0:
            labels.add("unsafe")
    except Exception:
        return set()
    return labels

def cost_fn(labels):
    """
    Example 0/1 cost: unsafe if the 'unsafe' predicate holds.
    """
    return 1.0 if "unsafe" in labels else 0.0

# --- 1.5) Or use default label_fn and cost_fn supplied by the environment (recommended)
from masa.env.tabular.media_streaming import label_fn, cost_fn

# --- 2) Build the environment and wrap in the correct order ---

env = MediaStreaming()

# Recommended: apply TimeLimit first (episode length is enforced before anything else).
env = TimeLimit(env, max_episode_steps=1_000)

# Attach labels to info["labels"] at every reset/step.
env = LabelledEnv(env, label_fn)

# Apply a constraint wrapper (example: cumulative cost/budget style constraint).
# Typical kwargs are shown; consult the constraint's docstring / Constraints API reference.
env = CumulativeCostEnv(env, cost_fn=cost_fn, budget=25.0)

# Finally, attach monitoring wrappers for constraints and reward logging.
env = ConstraintMonitor(env)
env = RewardMonitor(env)
```

## Random-Agent Interaction Loop (Gymnasium Style)

```python
num_episodes = 3

for ep in range(num_episodes):
    obs, info = env.reset(seed=ep)

    ep_return = 0.0
    ep_len = 0

    # Your monitors and constraints may attach additional fields; labels are always in info["labels"].
    labels = info.get("labels", set())
    print(f"[episode {ep}] reset labels={labels}")

    terminated = truncated = False
    step_cost = 0.0
    violated = False
    ep_cost = None
    ep_satisfied = None

    while not (terminated or truncated):
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        ep_return += float(reward)
        ep_len += 1

        labels = info.get("labels", set())

        # Common pattern: monitors expose step metrics in info.
        constraint = info.get("constraint", {})
        if isinstance(constraint, dict) and "step" in constraint:
            step_cost = constraint["step"].get("cost", 0.0)
            violated = constraint["step"].get("violated", False)

        if violated:
            print(f"  step={ep_len:04d} VIOLATION labels={labels} cost={step_cost}")

    # Episode-end metrics are often attached on the final transition by the monitors.
    constraint = info.get("constraint", {})
    if isinstance(constraint, dict) and "episode" in constraint:
        ep_cost = constraint["episode"].get("cum_cost", None)
        ep_satisfied = constraint["episode"].get("satisfied", None)

    print(
        f"[episode {ep}] return={ep_return:.2f} len={ep_len} "
        f"episode_cost={ep_cost} episode_satisfied={ep_satisfied}"
    )
```

## Training with PPO

Below is a minimal example showing how to initialize and train PPO using the wrapped environment. The specific PPO constructor and training API may include additional options such as logging, evaluation environments, or checkpointing.

```python
from masa.algorithms.ppo import PPO

# Optional: create a separate evaluation environment with the same wrapper stack.
def make_eval_env():
    eval_env = MediaStreaming()
    eval_env = TimeLimit(eval_env, max_episode_steps=1_000)
    eval_env = LabelledEnv(eval_env, label_fn)
    eval_env = CumulativeCostEnv(eval_env, cost_fn=cost_fn, budget=25.0)
    eval_env = ConstraintMonitor(eval_env)
    eval_env = RewardMonitor(eval_env)
    return eval_env

eval_env = make_eval_env()

algo = PPO(
    env,
    seed=0,
    device="auto",
    verbose=1,
    eval_env=eval_env,
    tensorboard_logdir=None,
)

algo.train(
    total_timesteps=200_000,
    num_eval_episodes=10,
    eval_freq=10_000,
    log_freq=2_000,
    stats_window_size=100,
)
```

## API Reference for `masa.common.utils.make_env`

::: masa.common.utils.make_env
    options:
      show_root_heading: false

## Next Steps

- [Constraints API Reference](../Common/Constraints) - View the common constraints supported by MASA.
