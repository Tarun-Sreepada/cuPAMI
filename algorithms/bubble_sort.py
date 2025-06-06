import os
import time
try:
    import psutil
except Exception:
    psutil = None

from .base_algorithm import BaseAlgorithm


class BubbleSort(BaseAlgorithm):
    def __init__(self, data=None):
        super().__init__()
        self.data = list(data) if data is not None else None
        self.sorted = None

    def run(self):
        start = time.time()
        if self.data is None:
            path = os.path.join('datasets', 'bubble_sort.txt')
            numbers = [int(x) for x in self.readfile(path).split()]
        else:
            numbers = list(self.data)

        n = len(numbers)
        for i in range(n):
            for j in range(0, n - i - 1):
                if numbers[j] > numbers[j + 1]:
                    numbers[j], numbers[j + 1] = numbers[j + 1], numbers[j]
        self.sorted = numbers
        self._runtime = time.time() - start

        if psutil:
            proc = psutil.Process(os.getpid())
            self._memory_rss = proc.memory_info().rss
            self._memory_uss = proc.memory_full_info().uss
        return numbers
