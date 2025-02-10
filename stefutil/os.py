"""
os-related
"""

import os
from os.path import join as os_join
from pathlib import Path
from typing import List, Union


__all__ = ['get_hostname', 'stem', 'rel_path']


def get_hostname() -> str:
    return os.uname().nodename


def stem(
        path: Union[str, Path], keep_ext=False, top_n: int = None, as_list: bool = False
) -> Union[str, List[str]]:
    """
    :param path: A potentially full path to a file
    :param keep_ext: If True, file extensions is preserved
    :param top_n: If given, keep the top `top_n` parent directories
    :param as_list: If True, return as a list
        Relevant only when `top_n` is given
    :return: The file name, without parent directories
    """
    if top_n:
        ret = stem(path=path, keep_ext=keep_ext, top_n=None)
        if isinstance(path, Path):
            dirs = []
            for _ in range(top_n):
                path = path.parent
                dirs.append(path.name)
            dirs.reverse()
        else:
            dirs = path.split(os.sep)
            dirs = dirs[-top_n-1:-1]
        dirs += [ret]
        return dirs if as_list else os_join(*dirs)
    else:
        # if it's a directory, always keep the extension
        if os.path.exists(path) and os.path.isdir(path):
            keep_ext = True
        return os.path.basename(path) if keep_ext else Path(path).stem


def rel_path(path: Union[str, Path], k: int = 3) -> str:
    """
    Syntactic sugar for a `stem` use case, keeping the top `k` parent dirs
    """
    return stem(path, top_n=k, keep_ext=True)


if __name__ == '__main__':
    def check_stem():
        n = 3
        # path = __file__
        path = '../dir-with.dot'
        path_ = Path(path)
        print(path)
        print(stem(path, top_n=n))
        print(stem(path, top_n=n, as_list=True))
        print(path_)
        print(stem(path_, top_n=n))
    # check_stem()

    def check_rel():
        n = 4
        path = __file__
        print(rel_path(path, k=n))
    check_rel()
