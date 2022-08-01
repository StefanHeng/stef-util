"""
primitive manipulation
"""

import re
import math
from typing import List, Any, Union

__all__ = ['nan', 'is_float', 'float_is_int', 'clean_whitespace', 'get_substr_indices']


nan = float('nan')


def is_float(x: Any, no_int=False, no_sci=False) -> bool:
    try:
        is_sci = isinstance(x, str) and 'e' in x.lower()
        f = float(x)
        is_int = f.is_integer()
        out = True
        if no_int:
            out = out and (not is_int)
        if no_sci:
            out = out and (not is_sci)
        return out
    except (ValueError, TypeError):
        return False


def float_is_int(f: float, eps: float = None) -> Union[int, bool]:
    if eps:
        return f.is_integer() or math.isclose(f, round(f), abs_tol=eps)
    else:
        return f.is_integer()


def clean_whitespace(s: str):
    if not hasattr(clean_whitespace, 'pattern_space'):
        clean_whitespace.pattern_space = re.compile(r'\s+')
    return clean_whitespace.pattern_space.sub(' ', s).strip()


def get_substr_indices(s: str, s_sub: str) -> List[int]:
    s_sub = re.escape(s_sub)
    return [m.start() for m in re.finditer(s_sub, s)]
