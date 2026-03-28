# Wrappers

Environment wrappers in MASA-Safe-RL are small, composable `gymnasium.Wrapper` utilities that:

1. preserve access to constraint-related objects through wrapper chains,
2. inject monitoring and metrics into `info`,
3. apply potential-based reward shaping for DFA-based constraints,
4. provide basic observation and reward normalization together with lightweight vector-environment helpers.

Key conventions:

- Constraint-enabled environments expose a `_constraint` object and often `label_fn` / `cost_fn` attributes. See `masa.common.constraints.base.BaseConstraintEnv`.
- Monitoring wrappers add structured dictionaries under `info["constraint"]` and or `info["metrics"]`.
- Vector wrappers in this file use a simple Python list API where observations, rewards, terminals, truncations, and infos are lists of length `VecEnvWrapperBase.n_envs`.

For potential-based shaping, the shaped cost inserted into `info` is:

$$
c'_t = c_t + \gamma \Phi(q_{t+1}) - \Phi(q_t),
$$

where $q_t$ is the DFA state, $c_t$ is the original constraint cost, $\Phi$ is the potential function, and $\gamma$ is the shaping discount factor.

## API Reference

### Base Classes

::: masa.common.wrappers.ConstraintPersistentWrapper
    options:
      show_root_heading: false

::: masa.common.wrappers.ConstraintPersistentObsWrapper
    options:
      show_root_heading: false

## Helpers

::: masa.common.wrappers.is_wrapped
    options:
      show_root_heading: false

::: masa.common.wrappers.get_wrapped
    options:
      show_root_heading: false

## Next Steps

- [Core Wrappers](Wrappers/Core%20Wrappers) - Core monitoring and time-limit wrappers.
- [Misc Wrappers](Wrappers/Misc%20Wrappers) - Miscellaneous shaping and observation wrappers.
- [Vectorized Envs](Wrappers/Vectorized%20Envs) - Synchronous vectorized environment helpers.
