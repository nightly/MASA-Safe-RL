# Metrics

## Overview

MASA metrics are designed to make it easy to record and report learning signals at different granularities: single scalars, streaming summary statistics, and approximate distributions. The logging stack is deliberately lightweight and backend-agnostic.

- **Scalars** are plain numeric values such as reward, loss, or episode length. Loggers keep a rolling window of recent scalar values and typically report a smoothed mean over that window.
- **Summary statistics** are handled by `masa.common.metrics.Stats`, which performs streaming aggregation over batches of values. It tracks running moments and extrema and exposes derived quantities such as the standard deviation:

$$
\sigma = \sqrt{\max(0, \mathbb{E}[X^2] - \mathbb{E}[X]^2)}.
$$

- **Distributions** are handled by `masa.common.metrics.Dist`, which maintains a fixed-size reservoir sample of a stream. This gives a compact approximation of the underlying distribution that can be plotted or logged as a histogram.
- **Logging** is performed by logger implementations, which accept mixtures of scalars, `Stats`, and `Dist` objects. They aggregate values over a configurable window and can emit to stdout and or TensorBoard with consistent key prefixing.

## API Reference

### Metrics Core

::: masa.common.metrics.Stats
    options:
      show_root_heading: false

::: masa.common.metrics.Dist
    options:
      show_root_heading: false

## Next Steps

- [Logging](Metrics/Logging) - Learn how logging is handled automatically in MASA.
