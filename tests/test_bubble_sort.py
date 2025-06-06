import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cProfile
import pstats

from algorithms.bubble_sort import BubbleSort


def test_bubble_sort_produces_sorted_list(tmp_path):
    prof_path = tmp_path / "profile.prof"
    alg = BubbleSort()
    cProfile.runctx('alg.run()', globals(), locals(), str(prof_path))

    stats = pstats.Stats(str(prof_path))
    stats.strip_dirs().sort_stats('cumtime').print_stats(5)

    assert alg.sorted == [1, 2, 4, 5, 8]
    assert alg.getruntime() is not None
    assert alg.getmemoryrss() is None or isinstance(alg.getmemoryrss(), int)
    assert alg.getmemoryuss() is None or isinstance(alg.getmemoryuss(), int)
