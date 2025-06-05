#!/usr/bin/env python3
"""Profile an algorithm with line_profiler and cProfile.

The profiling statistics are stored under ``results/<name>/``.  An HTML
visualization is produced with snakeviz when available.
"""

import importlib
import os
import sys
import cProfile
from line_profiler import LineProfiler

try:
    from snakeviz.cli import main as snakeviz_main
except Exception:  # snakeviz may not be installed
    snakeviz_main = None


def profile_algorithm(name: str) -> None:
    """Run profiling on the algorithm ``name``."""
    module = importlib.import_module(f"algs.{name}.{name}")
    class_name = ''.join(part.capitalize() for part in name.split('_'))
    Algorithm = getattr(module, class_name)
    alg = Algorithm()

    results_dir = os.path.join('results', name)
    os.makedirs(results_dir, exist_ok=True)
    line_path = os.path.join(results_dir, 'line_profile.lprof')
    cprof_path = os.path.join(results_dir, 'cprofile.prof')
    html_path = os.path.join(results_dir, 'profile.html')

    lp = LineProfiler()
    profiled_mine = lp(alg.mine)

    def run():
        profiled_mine()

    cProfile.runctx('run()', globals(), locals(), cprof_path)
    lp.dump_stats(line_path)

    if snakeviz_main:
        snakeviz_main([cprof_path, '--output-file', html_path])

    print(f'Saved line profile to {line_path}')
    print(f'Saved cProfile stats to {cprof_path}')
    if snakeviz_main:
        print(f'Saved snakeviz HTML to {html_path}')
    else:
        print('snakeviz not installed; HTML output was not generated')


def main() -> None:
    if len(sys.argv) != 2:
        print('Usage: python tests/profile_algorithm.py <name>')
        raise SystemExit(1)
    profile_algorithm(sys.argv[1])


if __name__ == '__main__':
    main()
