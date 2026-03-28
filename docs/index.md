# MASA Safe RL Docs

## Welcome

Welcome to MASA-Safe-RL, the Multi and Single Agent (MASA) Safe Reinforcement Learning library. The primary goal of this library is to develop a set of common constraints and environments for safe reinforcement learning research, built on top of the popular **[Gymnasium API](https://gymnasium.farama.org/)**. We span CMDPs, probabilistic constraints, reach-avoid, and LTL-safety (DFA) properties.

The library is in a very early stage of development and we greatly appreciate feedback from the community about what they would like to see implemented. MASA already provides a set of basic tabular algorithms for safe RL, but its main strength is a modular, reusable framework for building more expressive algorithms and constraints.

## Organization

The docs are organized as follows:

- **Get Started**: installation instructions and core concepts such as labelling functions, cost functions, and `make_env`.
- **Common API**: API references for constraints, wrappers, metrics logging, and temporal logics used in MASA.
- **Environments**: benchmark environments currently provided in MASA.
- **Algorithms**: algorithms currently provided in MASA.
- **Tutorials**: worked examples across the main MASA abstractions.
- **Misc**: auxiliary features such as probabilistic shielding.

We recommend following the docs in the provided order, starting from **Get Started**.

## Citation

If you use MASA in your research, please cite it in your publications.

```bibtex
@misc{Goodall2025MASASafeRL,
  title        = {{MASA-Safe-RL}: Multi and Single Agent Safe Reinforcement Learning},
  author       = {Goodall, Alexander W. and Adalat, Omar and Hamel De-le Court, Edwin and Belardinelli, Francesco},
  year         = {2025},
  howpublished = {\url{https://github.com/sacktock/MASA-Safe-RL/}},
  note         = {GitHub repository}
}
```

## Next Steps

- [Quick Start](Get%20Started/Quick%20Start) - Installation instructions for MASA.
