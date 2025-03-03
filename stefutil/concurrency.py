"""
concurrency

intended for (potentially heavy) data processing
"""


import os
import math
import heapq
import concurrent.futures
from typing import List, Tuple, Dict, Iterable, Callable, TypeVar, Union, Type

from tqdm.std import tqdm as std_tqdm  # root for type check
from tqdm.auto import tqdm
from tqdm.contrib import concurrent as tqdm_concurrent

from stefutil.container import group_n, length_hint
from stefutil.prettier import check_arg as ca, tqdc


__all__ = ['conc_map', 'batched_conc_map', 'conc_yield']


T = TypeVar('T')
K = TypeVar('K')

MapFn = Callable[[T], K]
BatchedMapFn = Callable[[Tuple[List[T], int, int]], List[K]]


def _check_conc_mode(mode: str):
    ca.assert_options('Concurrency Mode', mode, ['thread', 'process'])


def conc_map(
        fn: MapFn, args: Iterable[T], with_tqdm: Union[bool, Dict] = False, n_worker: int = os.cpu_count() - 1,
        mode: str = 'process'
) -> Iterable[K]:
    """
    Wrapper for `concurrent.futures.map`

    :param fn: A function
    :param args: A list of elements as input to the function
    :param with_tqdm: If true, progress bar is shown
        If dict, treated as `tqdm` concurrent kwargs
            note `chunksize` is helpful
    :param n_worker: Number of concurrent workers
    :param mode: One of ['thread', 'process']
        Function has to be pickleable if 'process'
    :return: Iterator of `lst` elements mapped by `fn` with thread concurrency
    """
    _check_conc_mode(mode=mode)
    if with_tqdm:
        cls = tqdm_concurrent.thread_map if mode == 'thread' else tqdm_concurrent.process_map
        tqdm_args = (isinstance(with_tqdm, dict) and with_tqdm) or dict()
        return cls(fn, args, max_workers=n_worker, **tqdm_args)
    else:
        cls = concurrent.futures.ThreadPoolExecutor if mode == 'thread' else concurrent.futures.ProcessPoolExecutor
        with cls(max_workers=n_worker) as executor:
            return executor.map(fn, args)


"""
classes instead of nested functions, pickleable for multiprocessing
"""


class Map:
    def __init__(self, fn, pbar=None):
        self.fn = fn
        self.pbar = pbar

    def __call__(self, x):
        ret = self.fn(x)
        if self.pbar:
            self.pbar.update(1)
        return ret


class BatchedMap:
    def __init__(self, fn, is_batched_fn: bool, pbar=None):
        self.fn = fn if is_batched_fn else Map(fn, pbar)
        self.pbar = pbar
        self.is_batched_fn = is_batched_fn

    def __call__(self, args):
        # adhere to single-argument signature for `conc_map`
        lst, s, e = args
        if self.is_batched_fn:
            ret = self.fn(lst[s:e])
            if self.pbar:
                # TODO: update on the element level may not give a good estimate of completion if too large a batch
                self.pbar.update(e-s + 1)
            return ret
        else:
            return [self.fn(lst[i]) for i in range(s, e)]


def batched_conc_map(
        fn: Union[MapFn, BatchedMapFn],
        args: Union[Iterable[T], List[T]], n: int = None, n_worker: int = os.cpu_count(),
        batch_size: int = None,
        with_tqdm: Union[bool, dict, tqdm] = False,
        is_batched_fn: bool = False,
        mode: str = 'thread'
) -> List[K]:
    """
    Batched concurrent mapping, map elements in list in batches
    Operates on batch/subset of `lst` elements given inclusive begin & exclusive end indices

    :param fn: A map function that operates on a single element
    :param args: A list of elements to map
    :param n: #elements to map if `it` is not Sized
    :param n_worker: Number of concurrent workers
    :param batch_size: Number of elements for each sub-process worker
        Inferred based on number of workers if not given
    :param with_tqdm: If true, progress bar is shown
        progress is shown on an element-level if possible
    :param is_batched_fn: If true, `conc_map` is called on the function passed in,
        otherwise, A batched version is created internally
    :param mode: One of ['thread', 'process']

    .. note:: Concurrently is not invoked if too little list elements given number of workers
        Force concurrency with `batch_size`
    """
    n = n or len(args)
    if (n_worker > 1 and n > n_worker * 4) or batch_size:  # factor of 4 is arbitrary, otherwise not worse the overhead
        preprocess_batch = batch_size or round(n / n_worker / 2)
        strts: List[int] = list(range(0, n, preprocess_batch))
        ends: List[int] = strts[1:] + [n]  # inclusive begin, exclusive end
        lst_out = []

        pbar = None
        if with_tqdm:
            tqdm_args = dict(mode=mode, n_worker=n_worker)
            if mode == 'thread':  # Able to show progress on element level
                # so create such a progress bar & disable for `conc_map`
                tqdm_args['with_tqdm'] = False
                if isinstance(with_tqdm, bool):
                    pbar = tqdm(total=n)
                elif isinstance(with_tqdm, dict):
                    _args = dict(total=n)
                    _args.update(with_tqdm)
                    pbar = tqdm(**_args)
                else:
                    assert isinstance(with_tqdm, std_tqdm)
                    pbar = with_tqdm
            else:  # `process`, have to rely on `tqdm.concurrent` which shows progress on batch level, see `conc_map`
                tqdm_args['with_tqdm'] = with_tqdm
        else:
            tqdm_args = dict(with_tqdm=False)

        batched_map = BatchedMap(fn, is_batched_fn, pbar)
        map_out = conc_map(fn=batched_map, args=[(args, s, e) for s, e in zip(strts, ends)], **tqdm_args)
        for lst_ in map_out:
            lst_out.extend(lst_)
        return lst_out
    else:
        gen = tqdm(args) if with_tqdm else args
        if is_batched_fn:
            _args = gen, 0, n
            return fn(*_args)
        else:
            return [fn(x) for x in gen]


class BatchedFn:
    def __init__(self, fn: MapFn = None, fn_keyword: str = None, pbar=None):
        self.fn = fn
        self.fn_keyword = fn_keyword
        self.pbar = pbar

    def __call__(self, args: Iterable[T]) -> List[K]:
        """
        No order enforced
        """
        ret = []
        for a in args:
            # ret.append(self.fn(a))
            ret.append(self.fn(**{self.fn_keyword: a}) if self.fn_keyword else self.fn(a))
            if self.pbar is not None:
                self.pbar.update(1)
        return ret


def conc_yield(
        fn: MapFn, args: Iterable[T], fn_kwarg: str = None, with_tqdm: Union[bool, Dict] = False, tqdm_class: Type[std_tqdm] = None,
        n_worker: int = os.cpu_count()-1,  # since the calling script consumes one process
        mode: str = 'process', batch_size: Union[int, bool] = None,
        process_chunk_size: int = None, process_chunk_multiplier: int = None, enforce_order: bool = False
) -> Iterable[K]:
    """
    Wrapper for `concurrent.futures`, yielding results as they become available, irrelevant of order
        Intended for loading up data where each instance takes relatively heavy processing time

    :param fn: A function
    :param args: A list of elements as input to the function
    :param fn_kwarg: If given, the keyword argument to pass to `fn`
    :param with_tqdm: If true, progress bar is shown
        If dict, treated as `tqdm` concurrent kwargs
            note `chunksize` is helpful
    :param tqdm_class: If given, use this tqdm class for the progress bar
        Defaults to the colored version
    :param n_worker: Number of concurrent workers
    :param mode: One of ['thread', 'process']
        Function has to be pickleable if 'process'
    :param batch_size: Number of elements for each sub-process worker
        Intended to lower concurrency overhead
    :param process_chunk_size: If mode is `process`,
        elements are chunked into groups of `process_chunk_size` for a sequence of concurrent pools
    :param process_chunk_multiplier: If mode is `process`,
        elements are chunked into groups of `n_worker` x `batch_size` x `process_chunk_multiplier`
            for a sequence of concurrent pools
    :param enforce_order: If true, results are yielded in the order of args passed
    :return: Iterator of `lst` elements mapped by `fn` with thread concurrency
    """
    _check_conc_mode(mode=mode)

    is_thread = mode == 'thread'
    cls = concurrent.futures.ThreadPoolExecutor if is_thread else concurrent.futures.ProcessPoolExecutor
    executor = cls(max_workers=n_worker)

    pbar = None
    if with_tqdm:
        if not isinstance(pbar, std_tqdm):
            n = _get_length(args)
            if n:
                if batch_size:
                    n = math.ceil(n / batch_size)
            else:
                n = None
            tqdm_args = dict(total=n)

            if isinstance(with_tqdm, dict):
                tqdm_args.update(with_tqdm)
            tqdm_class = tqdm_class or tqdc
            pbar = tqdm_class(**tqdm_args)

    if batch_size:
        batch_size = 32 if isinstance(batch_size, bool) else batch_size
        # pbar doesn't work w/ pickle hence multiprocessing
        fn = BatchedFn(fn=fn, fn_keyword=fn_kwarg, pbar=pbar if is_thread else None)
        if process_chunk_size is not None:
            chunk = process_chunk_size
        elif process_chunk_multiplier is not None:
            chunk = n_worker * batch_size * process_chunk_multiplier
        else:
            chunk = None

        if not is_thread and chunk:
            it_args = group_n(args, chunk)
            for args_ in it_args:
                futures = {executor.submit(fn, args_): i for i, args_ in enumerate(group_n(args_, batch_size))}

                pq, index = ([], 0) if enforce_order else (None, None)

                for f in concurrent.futures.as_completed(futures):
                    res = f.result()

                    if enforce_order:
                        heapq.heappush(pq, (futures[f], res))
                        while pq and pq[0][0] == index:
                            _, res = heapq.heappop(pq)
                            yield from res
                            index += 1

                            if with_tqdm:
                                pbar.update(len(res))

                    else:
                        if with_tqdm:
                            pbar.update(len(res))
                        yield from res

                    del futures[f]
        else:
            futures = {executor.submit(fn, args_): i for i, args_ in enumerate(group_n(args, batch_size))}

            pq, index = ([], 0) if enforce_order else (None, None)

            for f in concurrent.futures.as_completed(futures):
                res = f.result()

                if enforce_order:
                    heapq.heappush(pq, (futures[f], res))
                    while pq and pq[0][0] == index:
                        _, res = heapq.heappop(pq)
                        yield from res
                        index += 1

                        if not is_thread and with_tqdm:
                            pbar.update(len(res))

                else:
                    if not is_thread and with_tqdm:
                        pbar.update(len(res))
                    yield from res

                del futures[f]

    else:
        if fn_kwarg:
            futures = {executor.submit(fn, **{fn_kwarg: a}): i for i, a in enumerate(args)}
        else:
            futures = {executor.submit(fn, a): i for i, a in enumerate(args)}

        if enforce_order:
            pq = []  # priority queue to hold (index, result) pairs
            index = 0  # the next index to yield
        else:
            pq, index = None, None

        for f in concurrent.futures.as_completed(futures):  # TODO: process chunking
            res = f.result()

            if enforce_order:
                heapq.heappush(pq, (futures[f], res))  # prioritize by index
                while pq and pq[0][0] == index:
                    _, res = heapq.heappop(pq)
                    yield res
                    index += 1

                    if with_tqdm:
                        pbar.update(1)
            else:
                if with_tqdm:
                    pbar.update(1)
                yield res
            
            del futures[f]


if __name__ == '__main__':
    from stefutil.prettier import rich_progress, rich_console_log

    import time
    import random

    from stefutil.prettier import style, icecream as sic, get_logger

    logger = get_logger('Conc Dev')

    def _work(task_idx: int = None):
        t = round(random.uniform(0.5, 3), 3)
        print(f'Task {style(task_idx)} launched, will sleep for {style(t)} s')
        time.sleep(t)

        print(f'Task {style(task_idx)} is done')
        return task_idx

    def _dummy_fn(xx: int = None):
        x = xx
        t_ms = random.randint(500, 3000)

        # logger.info(f'Calling dummy_fn w/ arg {s.i(x)}, will sleep for {s.i(t_ms)}ms')
        time.sleep(t_ms / 1000)

        return x, [random.random() for _ in range(int(1e6))]

    def try_concurrent_yield():
        # this will gather all results and return
        # with ThreadPoolExecutor() as executor:
        #     for result in executor.map(work, range(3)):
        #         print(result)
        # executor = concurrent.futures.ThreadPoolExecutor()
        executor = concurrent.futures.ProcessPoolExecutor()
        args = list(range(4))
        futures = [executor.submit(_work, a) for a in args]
        for f in concurrent.futures.as_completed(futures):
            res = f.result()
            sic(res)
    # try_concurrent_yield()

    def check_conc_map():
        n = 10
        # for res in conc_map(fn=_work, args=range(n), with_tqdm=dict(total=n), n_worker=4, mode='process'):
        #     sic(res)
        from stefutil.prettier import rich_console_log
        for res in rich_progress(conc_map(fn=_work, args=range(n), mode='process'), total=n):
            rich_console_log(res)
    # check_conc_map()

    def check_conc_yield():
        batch = False

        if batch:
            bsz = 4
            n = 100
        else:
            bsz = None
            n = 10

        # it = range(n)
        it = enumerate(range(n))
        with_tqdm = dict(total=n)
        # with_tqdm = True
        n_worker = 4

        # mode = 'process'
        mode = 'thread'
        for res in conc_yield(fn=_work, args=it, with_tqdm=with_tqdm, n_worker=n_worker, mode=mode, batch_size=bsz):
            sic(res)
    check_conc_yield()

    def check_conc_mem_use():
        from concurrent.futures import ThreadPoolExecutor, as_completed

        n = 30
        # remove_job = True
        remove_job = False

        # test_code = True
        test_code = False
        if test_code:
            n_processed = 0
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {executor.submit(_dummy_fn, i) for i in range(n)}
                for f in as_completed(futures):
                    i, _ = f.result()
                    if remove_job:
                        futures.remove(f)
                    n_processed += 1
                    logger.info(f'Process {style(i)} terminated, {style(n_processed)} / {style(n)} processed')
        else:
            args = dict(with_tqdm=dict(total=n), n_worker=3, mode='process', batch_size=4)
            for i in conc_yield(fn=_dummy_fn, args=range(n), **args):
                i, _ = i
                logger.info(f'Process {style(i)} terminated')
    # check_conc_mem_use()

    def check_conc_process_chunk():
        # n = 200
        # n = 100
        n = 50
        # chunk_mult = 4
        # bsz, n_worker = 4, 4
        bsz = 4
        # bsz = False
        # args = dict(n_worker=4, mode='process', batch_size=bsz, process_chunk_size=chunk_mult * bsz * n_worker)
        args = dict(n_worker=4, mode='process', batch_size=bsz, process_chunk_multiplier=None)

        # for i in conc_yield(fn=_dummy_fn, fn_kwarg='xx', args=range(n), with_tqdm=dict(total=n), **args):
        #     i, _ = i
        #     logger.info(f'Process {s.i(i)} terminated')

        lst = []
        for i in rich_progress(conc_yield(fn=_dummy_fn, fn_kwarg='xx', args=range(n), **args, enforce_order=True), total=n):
            i, _ = i
            # logger.info(f'Process {s.i(i)} terminated')
            lst.append(i)
        rich_console_log(lst)
    # check_conc_process_chunk()
