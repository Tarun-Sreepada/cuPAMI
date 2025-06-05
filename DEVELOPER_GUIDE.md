# Developer Guide

This document explains how to contribute new algorithms and documentation to **cuPAMI**.

## Creating an algorithm skeleton

Use the helper script to scaffold directories for a new algorithm:

```bash
python utils/create_algorithm.py <name> [--lang python|cpp|cuda]
```

This command creates folders under `algs/`, `datasets/`, `experiments/` and `tests/`
with placeholder files. Choose `--lang` to generate a pure Python implementation
or wrappers for C++/CUDA code.

## Adding tests

Place unit tests for your algorithm inside `tests/<name>/`.  The repository uses
`pytest` so add standard test modules and run:

```bash
pytest -q
```

## Documenting your algorithm

Algorithms documented with Python docstrings are automatically included in the
Sphinx API reference.  Add descriptive docstrings to your modules and classes.

## Building the documentation

After adding docstrings or editing the documentation sources under `docs/`,
rebuild the HTML documentation with:

```bash
make -C docs html
```

The generated site is written to `docs/_build/html`.
