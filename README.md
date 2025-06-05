# cuPAMI

This repository contains GPU and CPU implementations for various pattern
mining algorithms.
Algorithms live under the `algs/` directory and helper scripts can be found in
`utils/`.  Datasets should be placed in `datasets/` and notebooks for
experiments reside in `experiments/`.  Any unit tests live under `tests/`.

To create a new algorithm skeleton with matching experiment, dataset and test
folders run:

```bash
python utils/create_algorithm.py <name>
```

## Base algorithm interface

A default base class is provided in `algs/base_algorithm.py`.  New
implementations should extend this class and populate runtime and memory
statistics during execution.  The class exposes three common accessors:

* `getRuntime()` – return the runtime of the last `mine()` call.
* `getMemoryRSS()` – return the resident set size (RSS) memory usage in bytes.
* `getMemoryUSS()` – return the unique set size (USS) memory usage in bytes.

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
