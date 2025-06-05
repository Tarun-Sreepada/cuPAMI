# cuPAMI

This repository contains GPU and CPU implementations for various pattern mining algorithms.
Algorithms live under the `algs/` directory and helper scripts can be found in
`utils/`.  Datasets should be placed in `datasets/` and notebooks for
experiments reside in `experiments/`.  Any unit tests live under `tests/`.

To create a new algorithm skeleton with matching experiment, dataset and test
folders run:

```bash
python utils/create_algorithm.py <name> [--lang python|cpp|cuda]
```
The optional `--lang` flag generates either a pure Python implementation or a
wrapper with C++/CUDA source under `algs/<name>/src/` that can be compiled
separately.  The default is `python`.

## Installation

Install the project dependencies with:

```bash
pip install -r requirements.txt
```

Some algorithms rely on GPU libraries such as `cupy` and `cudf`. Ensure your
environment satisfies their prerequisites.

## Base algorithm interface

A default base class is provided in `algs/base_algorithm.py`.  New
implementations should extend this class and populate runtime and memory
statistics during execution.  The class exposes three common accessors:

* `getRuntime()` – return the runtime of the last `mine()` call.
* `getMemoryRSS()` – return the resident set size (RSS) memory usage in bytes.
* `getMemoryUSS()` – return the unique set size (USS) memory usage in bytes.
* `readFile(path)` – return the contents of a file for custom processing.

```python
from algs.base_algorithm import BaseAlgorithm

class MyMiner(BaseAlgorithm):
    def mine(self):
        start = time.time()
        # ... algorithm implementation ...
        self.runtime = time.time() - start
        self.memoryRSS = psutil.Process(os.getpid()).memory_info().rss
        self.memoryUSS = psutil.Process(os.getpid()).memory_full_info().uss
```

These accessors make it easy to benchmark algorithms in a consistent manner.

## Profiling algorithms

A helper script is provided to profile algorithms using both `line_profiler` and
`cProfile`.  The results are written to the `results/` directory and an HTML
report is generated via `snakeviz` when available.

```bash
python tests/profile_algorithm.py <name>
```

Replace `<name>` with the name of the algorithm you wish to profile.  The script
will create a folder under `results/<name>/` containing `line_profile.lprof`,
`cprofile.prof` and `profile.html`.

## Building documentation

Documentation is generated with [Sphinx](https://www.sphinx-doc.org/). To build
the HTML docs run:

## Building documentation

Documentation is generated with [Sphinx](https://www.sphinx-doc.org/). To build the HTML docs run:

```bash
make -C docs html
```

The API reference is produced automatically from the modules under `algs/`. New
algorithms documented with Python docstrings will appear in the generated
documentation.

For more details on contributing algorithms and tests, see
[DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md).
The API reference is produced automatically from the modules under `algs/`. New algorithms documented with Python docstrings will appear in the generated documentation.
