"""
function-level manipulation
"""
import os
from os.path import join as os_join
import sys
import time
from typing import Callable, Union

from stefutil.prettier import *


__all__ = ['profile_runtime', 'RecurseLimit']


def profile_runtime(
        callback: Callable, sleep: Union[float, int] = None, write: bool = False, output_dir_name: str = None, output_file_name: str = None
):
    import cProfile
    import pstats
    profiler = cProfile.Profile()
    profiler.enable()
    callback()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    if sleep:    # Sometimes, the top rows in `print_states` are now shown properly
        time.sleep(sleep)
    stats.print_stats()

    if write:
        now_ = now(for_path=True, fmt='short-full')
        fnm = f'{now_}_RuntimeProfile'
        if output_file_name:
            fnm = f'{fnm}_{output_file_name}'
        if output_dir_name:
            os.makedirs(output_dir_name, exist_ok=True)
            fnm = os_join(output_dir_name, fnm)
        stats.dump_stats(f'{fnm}.log')


class RecurseLimit:
    # credit: https://stackoverflow.com/a/50120316/10732321
    def __init__(self, limit):
        self.limit = limit

    def __enter__(self):
        self.old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(self.limit)

    def __exit__(self, kind, value, tb):
        sys.setrecursionlimit(self.old_limit)
