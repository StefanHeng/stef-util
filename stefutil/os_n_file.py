"""
os-related
"""

import os
import pathlib
from typing import Union


__all__ = ['get_hostname', 'stem']


def get_hostname() -> str:
    return os.uname().nodename


def stem(path_: Union[str, pathlib.Path], ext=False) -> str:
    """
    :param path_: A potentially absolute path to a file
    :param ext: If True, file extensions is preserved
    :return: The file name, without parent directories
    """
    return os.path.basename(path_) if ext else pathlib.Path(path_).stem
