# Probabilistic Computation Tree Logic (PCTL)

## Overview

MASA supports bounded Probabilistic Computation Tree Logic (PCTL) for reasoning about probabilistic safety and liveness properties of Markov decision processes and the Markov chains induced by fixing a policy.

The implementation focuses on:

- bounded temporal operators for finite-horizon reasoning,
- vectorized evaluation of atomic predicates via a labelling function,
- JAX-accelerated computation for scalable probability-sequence evaluation,
- dense and compact transition-kernel representations for both small tabular and large sparse dynamics.

## Bounded PCTL Formulae in MASA

MASA implements a bounded fragment of PCTL: all temporal operators are evaluated over a finite horizon, which aligns naturally with finite rollouts and Monte Carlo estimation.

A bounded PCTL formula is represented as a tree of `BoundedPCTLFormula` objects and falls into two broad categories.

### State (Propositional) Formulae

State formulae are Boolean formulae evaluated at a single state, without reference to transitions. MASA supports:

- `Truth`
- atomic predicates via `Atom`
- Boolean connectives: $\neg$, $\land$, $\lor$, and $\to`

State formulae are evaluated using a vectorized labeling representation:

- A `LabelFn` maps each state to a set of atomic predicates.
- Internally, MASA builds a dense matrix:

$$
\mathrm{vec\_label\_fn} \in \{0, 1\}^{|AP| \times |S|}
$$

where `vec_label_fn[i, s] = 1` if atomic predicate `AP[i]` holds in state `s`.

State formulae have zero temporal bound, so they do not extend the horizon.

### Probabilistic Temporal Formulae

Temporal formulae reason about the probability that events occur along paths within a bounded number of steps. MASA supports the following bounded operators.

#### Next

$$
\mathbb{P}_{\ge p}[\,X\ \Phi\,]
$$

The formula holds in state `s` if the probability that $\Phi$ holds in the next state is at least `p`.

#### Until (Bounded)

$$
\mathbb{P}_{\ge p}[\,\Phi_1\ U^{\le B}\ \Phi_2\,]
$$

The formula holds in state `s` if the probability that $\Phi_2$ becomes true within `B` steps while $\Phi_1$ holds at all preceding steps is at least `p`.

#### Eventually (Syntactic Sugar)

$$
\mathbb{P}_{\ge p}[\,F^{\le B}\ \Phi\,] \equiv
\mathbb{P}_{\ge p}[\,\mathrm{True}\ U^{\le B}\ \Phi\,]
$$

#### Always (Duality)

$$
\mathbb{P}_{\ge p}[\,G^{\le B}\ \Phi\,] \equiv
\neg\,\mathbb{P}_{\geq 1-p}[\,\mathrm{True}\ U^{\le B}\ \neg\Phi\,]
$$

#### Probability Sequences and Bounds

Each probabilistic operator contributes a local bound and computes a per-state probability sequence up to that horizon:

$$
P_0(s), P_1(s), \dots, P_B(s)
$$

where $P_k(s)$ is the probability that the subformula's path condition is satisfied within $k$ steps starting from state $s$.

Satisfaction for probabilistic operators is obtained by thresholding the probability at the operator's bound:

$$
\text{sat}(s) = \mathbf{1}\{P_B(s) \ge p\}
$$

## Transition Kernels and Policy-Induced Chains

PCTL evaluation in MASA is performed on a Markov chain kernel. For an MDP, MASA first fixes a policy and collapses the MDP into a Markov chain by taking expectations over actions.

MASA supports two representations of the resulting transition kernel.

### Dense Kernel

A dense Markov-chain transition matrix of shape `(n_states, n_states)`. Entry `(s', s)` gives the probability of transitioning from `s` to `s'`, so columns represent source states.

### Compact Kernel

A sparse successor representation that stores at most `K` successors per state:

- `succ` with shape `(K, n_states)` storing successor state ids
- `p` with shape `(K, n_states)` storing the corresponding transition probabilities

In both cases, probabilistic operators such as `Next` and bounded `Until` evaluate expectations of the form $\mathbb{E}[V(s')]$ by either matrix multiplication for dense kernels or successor-index gathering and weighted sums for compact kernels.

## Model Checking in MASA

MASA provides both exact and sampling-based checking behind a shared interface:

- If a transition model is available, MASA can evaluate the probability sequences deterministically by dynamic programming over the policy-induced Markov chain.
- If the model is unknown or too large, MASA can estimate satisfaction probabilities by Monte Carlo sampling of bounded trajectories under the policy and then compare the estimate to the formula's probability threshold.

Pure state formulae such as `Truth`, `Atom`, and Boolean connectives can be evaluated directly without sampling, while probabilistic temporal operators use either the exact recurrence or sampled trajectory satisfaction depending on the chosen approach.

## Next Steps

- [Propositional Operators](PCTL/Propositional%20Operators) - API reference for PCTL propositional operators.
- [Temporal Operators](PCTL/Temporal%20Operators) - API reference for PCTL temporal operators.
- [Model Checking](PCTL/Model%20Checking) - API reference for PCTL model checking.
