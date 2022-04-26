"""
concurrency

intended for (potentially heavy) data processing
"""

import os
import concurrent.futures
from typing import List, Iterable, Callable, TypeVar, Union

from tqdm import tqdm


__all__ = ['conc_map', 'batched_conc_map']


T = TypeVar('T')
K = TypeVar('K')


def conc_map(fn: Callable[[T], K], it: Iterable[T], with_tqdm=False) -> Iterable[K]:
    """
    Wrapper for `concurrent.futures.map`

    :param fn: A function
    :param it: A list of elements
    :return: Iterator of `lst` elements mapped by `fn` with concurrency
    :param with_tqdm: If true, progress bar is shown
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        ret = list(tqdm(executor.map(fn, it), total=len(list(it)))) if with_tqdm else executor.map(fn, it)
    return ret


def batched_conc_map(
        # fn: Callable[[Tuple[List[T], int, int]], K],
        fn: Callable[[T], K],
        lst: List[T], n_worker: int = os.cpu_count(),
        batch_size: int = None,
        with_tqdm: Union[bool, tqdm] = False  # TODO: doesn't seem to work as expected
) -> List[K]:
    """
    Batched concurrent mapping, map elements in list in batches

    :param fn: A map function that operates on a single element
        A version that operates on batch/subset of `lst` elements given inclusive begin & exclusive end indices
            is created internally
    :param lst: A list of elements to map
    :param n_worker: Number of concurrent workers
    :param batch_size: Number of elements for each sub-process worker
        Inferred based on number of workers if not given
    :param with_tqdm: If true, progress bar is shown

    .. note:: Concurrently is not invoked if too little list elements given number of workers
        Force concurrency with `batch_size`
    """
    n: int = len(lst)
    if (n_worker > 1 and n > n_worker * 4) or batch_size:  # factor of 4 is arbitrary, otherwise not worse the overhead
        preprocess_batch = batch_size or round(n / n_worker / 2)
        strts: List[int] = list(range(0, n, preprocess_batch))
        ends: List[int] = strts[1:] + [n]  # inclusive begin, exclusive end
        lst_out = []

        pbar = None
        if with_tqdm:
            pbar = tqdm(total=len(lst)) if with_tqdm is True else with_tqdm

        if with_tqdm:
            def map_single(x):
                ret = fn(x)
                pbar.update(1)
                return ret
        else:
            map_single = fn

        def batched_map(fnms_, s, e):
            return [map_single(fnms_[i]) for i in range(s, e)]

        # Expand the args
        map_out = conc_map(
            lambda args_: batched_map(*args_), [(lst, s, e) for s, e in zip(strts, ends)], with_tqdm=False
        )
        for lst_ in map_out:
            lst_out.extend(lst_)
        return lst_out
    else:
        # args = lst, 0, n
        # return fn(*args)
        gen = tqdm(lst) if with_tqdm else lst
        return [fn(x) for x in gen]
