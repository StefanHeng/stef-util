"""
concurrency

intended for (potentially heavy) data processing
"""

import os
import concurrent.futures
from typing import List, Tuple, Iterable, Callable, TypeVar, Union

import numpy as np
from tqdm import tqdm


__all__ = ['conc_map', 'batched_conc_map']


T = TypeVar('T')
K = TypeVar('K')

Map = Callable[[T], K]
BatchedMap = Callable[[Tuple[List[T], int, int]], List[K]]


def conc_map(fn: Map, it: Iterable[T], with_tqdm=False) -> List[K]:
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
        fn: Union[Map, BatchedMap],
        lst: Union[List[T], np.array], n_worker: int = os.cpu_count(),
        batch_size: int = None,
        with_tqdm: Union[bool, tqdm] = False,
        is_batched_fn: bool = False,
) -> List[K]:
    """
    Batched concurrent mapping, map elements in list in batches
    Operates on batch/subset of `lst` elements given inclusive begin & exclusive end indices

    :param fn: A map function that operates on a single element
    :param lst: A list of elements to map
    :param n_worker: Number of concurrent workers
    :param batch_size: Number of elements for each sub-process worker
        Inferred based on number of workers if not given
    :param with_tqdm: If true, progress bar is shown
        progress is shown on an element-level if possible
    :param is_batched_fn: If true, `conc_map` is called on the function passed in,
        otherwise, A batched version is created internally

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

        if is_batched_fn:
            def batched_map(lst__, s, e):
                ret = fn(lst__[s:e])
                if pbar:
                    # update on an element level may not give a good estimate of completion if too large a batch
                    pbar.update(e-s + 1)
                return ret
        else:
            if with_tqdm:
                def map_single(x):
                    ret = fn(x)
                    pbar.update(1)
                    return ret
            else:
                map_single = fn

            def batched_map(fnms_, s, e):
                return [map_single(fnms_[i]) for i in range(s, e)]
        map_out = conc_map(  # Expand the args
            lambda args_: batched_map(*args_), [(lst, s, e) for s, e in zip(strts, ends)], with_tqdm=False
        )
        for lst_ in map_out:
            lst_out.extend(lst_)
        return lst_out
    else:
        gen = tqdm(lst) if with_tqdm else lst
        if is_batched_fn:
            args = gen, 0, n
            return fn(*args)
        else:
            return [fn(x) for x in gen]
