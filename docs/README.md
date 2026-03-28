# MASA Docs

This folder contains the documentation for [MASA-Safe-RL](https://github.com/sacktock/MASA-Safe-RL/)

## Building

The docs can be built with:
```sh
uv sync --group docs
uv run zensical build
```

The generated site is written to `site/`.

Live reloading is supported with either:
```sh
uv sync --group docs
uv run zensical serve
```

The local preview is served at `http://127.0.0.1:8000` by default.