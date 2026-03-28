# Constraints

MASA constraints are stateful monitors layered on top of labelled environments.

- `CostFn` is the core callable type for turning a label set into a scalar cost.
- `Constraint` defines the monitor protocol used throughout MASA.
- `BaseConstraintEnv` is the Gymnasium wrapper base class that updates a constraint from `info["labels"]` and exposes constraint metrics through the environment interface.

## API Reference

::: masa.common.constraints.base.Constraint
    options:
      show_root_heading: false

::: masa.common.constraints.base.BaseConstraintEnv
    options:
      show_root_heading: false

## Next Steps

- [CMDP](Constraints/Constrained%20Markov%20Decision%20Process%20(CMDP)) - Budgeted constrained MDPs.
- [LTL Safety](Constraints/LTL%20Safety) - Safety LTL as a monitor and constraint.
- [PCTL](Constraints/PCTL) - A probabilistic computation tree logic constraint.
- [Step-wise Probabilistic](Constraints/Stepwise%20Probabilistic) - Undiscounted probabilistic step-wise safety constraints.
- [Reach Avoid](Constraints/Reach%20Avoid) - Reach-avoid constraints.
- [Multi Agent](Constraints/Multi%20Agent) - Multi-agent constraint models such as ATL.
