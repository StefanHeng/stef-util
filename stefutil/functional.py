"""
function-level manipulation
"""
import os
import sys
import time
from os.path import join as os_join
from typing import Callable, Union, Any

from stefutil.prettier import ca, date


__all__ = ['profile_runtime', 'RecurseLimit']


def profile_runtime(
        callback: Callable, sleep: Union[float, int] = None, mode: str = 'cumulative', interval: float = 0.001,
        disable_stdout: bool = False, write: bool = False, output_dir_name: str = None, output_file_name: str = None
) -> Any:
    """
    :param callback: The function to profile
    :param sleep: Sleep for a moment before printing the stats
    :param mode: profiling mode, one of [`cumulative`, `call-stack`]
        If cumulative, cumulative runtime profiling w/ python native `cProfile`
        If call-stack, call-stack profiling w/ `pyinstrument`
    :param interval: The profiling interval for `pyinstrument`
    :param disable_stdout: Whether to disable stdout printing
    :param write: Whether to write the profiling stats to a file
    :param output_dir_name: The directory to write the profiling stats
    :param output_file_name: The filename to write the profiling stats
    """
    ca.assert_options(display_name='Profile Mode', val=mode, options=['cumulative', 'call-stack'])

    stats = None
    if mode == 'cumulative':
        import pstats
        import cProfile

        profiler = cProfile.Profile()
        profiler.enable()
        callback()
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumtime')
    else:
        import pyinstrument
        profiler = pyinstrument.Profiler(interval=interval)
        profiler.start()
        callback()
        profiler.stop()
    if sleep:    # Sometimes, the top rows in `print_states` are now shown properly
        time.sleep(sleep)

    if not disable_stdout:
        if mode == 'cumulative':
            stats.print_stats()
        else:
            profiler.print(color=True)
    if write or output_dir_name or output_file_name:
        now_ = date()
        fnm = 'Cumulative-Runtime-Profile' if mode == 'cumulative' else 'Call-Stack-Profile'
        fnm = f'{now_}_{fnm}'
        if output_file_name:
            fnm = f'{fnm}_{output_file_name}'
        if output_dir_name:
            os.makedirs(output_dir_name, exist_ok=True)
            fnm = os_join(output_dir_name, fnm)

        if mode == 'cumulative':
            with open(f'{fnm}.log', 'w') as f:
                stats.stream = f
                stats.print_stats()
        else:
            profiler.write_html(path=f'{fnm}.html')
    return stats


class RecurseLimit:
    # credit: https://stackoverflow.com/a/50120316/10732321
    def __init__(self, limit):
        self.limit = limit

    def __enter__(self):
        self.old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(self.limit)

    def __exit__(self, kind, value, tb):
        sys.setrecursionlimit(self.old_limit)


if __name__ == '__main__':
    def test_profile_runtime():
        def fib(n):
            if n <= 1:
                return n
            return fib(n - 1) + fib(n - 2)
        # md = 'cumulative'
        md = 'call-stack'
        profile_runtime(lambda: fib(5), mode=md, write=True)
    test_profile_runtime()
