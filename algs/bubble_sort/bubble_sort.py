from __future__ import annotations

import os
import time
from typing import Iterable, List, Optional

try:
    import psutil
except Exception:  # pragma: no cover - psutil may not be installed
    psutil = None

from algs.base_algorithm import BaseAlgorithm


class BubbleSort(BaseAlgorithm):
    """Simple bubble sort implementation.

    Parameters
    ----------
    data : Iterable[int], optional
        Sequence of integers to sort. If omitted, numbers are read from
        ``datasets/bubble_sort/bubble_sort.txt``.

    Attributes
    ----------
    sorted : list[int]
        The sorted output produced by :py:meth:`mine`.
    """

    def __init__(self, data: Optional[Iterable[int]] = None) -> None:
        super().__init__()
        self.data = list(data) if data is not None else None
        self.sorted: Optional[List[int]] = None

    def mine(self) -> List[int]:
        """Sort the input data using bubble sort.

        Returns
        -------
        list[int]
            The sorted numbers.
        """
        start = time.time()

        if self.data is None:
            path = os.path.join('datasets', 'bubble_sort', 'bubble_sort.txt')
            contents = self.readFile(path)
            arr = [int(x) for x in contents.split()]
        else:
            arr = list(self.data)

        n = len(arr)
        for i in range(n):
            for j in range(0, n - i - 1):
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]

        self.sorted = arr
        self.runtime = time.time() - start
        if psutil:
            proc = psutil.Process(os.getpid())
            self.memoryRSS = proc.memory_info().rss
            self.memoryUSS = proc.memory_full_info().uss
        else:  # pragma: no cover - psutil may be missing
            self.memoryRSS = None
            self.memoryUSS = None
        return arr
