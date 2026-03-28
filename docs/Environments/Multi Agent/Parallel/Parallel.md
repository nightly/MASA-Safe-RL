# Parallel Multi Agent Environments

PettingZoo supports two main types of environment interfaces:

- `Parallel`, where each agent selects an action independently at each timestep. These environments are naturally representable as partially observable stochastic games.
- `AEC`, for turn-based environments using the agent-environment cycle model. Parallel environments can be converted to AEC environments when needed.

## Next Steps

- [Markov Stag Hunt](Gridworlds/Markov%20Stag%20Hunt) - A parallel gridworld benchmark.
- [Prisoners Dilemma](Matrix%20Games/Prisoners%20Dilemma) - A classic matrix game.
- [Battle of the Sexes](Matrix%20Games/Battle%20of%20the%20Sexes) - A coordination matrix game.
