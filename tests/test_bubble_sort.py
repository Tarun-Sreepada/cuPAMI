import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from algs.bubble_sort.bubble_sort import BubbleSort


def test_sort_inline():
    alg = BubbleSort([5, 1, 4, 2, 8])
    result = alg.mine()
    assert result == [1, 2, 4, 5, 8]
    assert alg.getRuntime() is not None
    # memory metrics may be None if psutil is unavailable
    assert alg.getMemoryRSS() is None or isinstance(alg.getMemoryRSS(), int)


def test_sort_from_file():
    alg = BubbleSort()
    path = os.path.join('datasets', 'bubble_sort', 'bubble_sort.txt')
    assert os.path.exists(path)
    result = alg.mine()
    assert result == [1, 2, 4, 5, 8]
