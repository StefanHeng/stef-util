"""
enhanced-built-in function
"""

import random
from typing import Union


__all__ = ['vars_', 'get_random_generator']


def vars_(obj, include_private=False):
    """
    :return: A variant of `vars` that returns all properties and corresponding values in `dir`, except the
    generic ones that begins with `_`
    """
    def is_relevant():
        if include_private:
            return lambda a: not a.startswith('__')
        else:
            return lambda a: not a.startswith('__') and not a.startswith('_')
    attrs = filter(is_relevant(), dir(obj))
    return {a: getattr(obj, a) for a in attrs}


def get_random_generator(generator: Union[int, random.Random] = None) -> random.Random:
    """
    Get a random generator
    """
    if isinstance(generator, random.Random):
        return generator
    elif generator is not None and isinstance(generator, int):
        return random.Random(generator)
    else:
        return random.Random()  # effectively no seed
