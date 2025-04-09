"""
primitive manipulation
"""


import re
import math
from typing import List, Any, Union


__all__ = [
    'nan',
    'is_int', 'float_is_sci', 'is_float', 'float_is_int', 'is_number',
    'clean_whitespace', 'get_substr_indices'
]


nan = float('nan')


def is_int(x: Any, allow_str: bool = False) -> bool:
    if allow_str and isinstance(x, str):
        try:
            x = int(x)
        except ValueError:
            return False
    return isinstance(x, int) or (isinstance(x, float) and x.is_integer())


def float_is_sci(f: Union[float, str]) -> bool:
    return 'e' in str(f).lower()


def is_float(x: Any, no_int=False, no_sci=False) -> bool:
    try:
        f = float(x)
        is_int_ = f.is_integer()
        out = True
        if no_int:
            out = out and (not is_int_)
        if no_sci:
            out = out and (not float_is_sci(x))
        return out
    except (ValueError, TypeError):
        return False


def float_is_int(f: float, eps: float = None) -> Union[int, bool]:
    if eps:
        return f.is_integer() or math.isclose(f, round(f), abs_tol=eps)
    else:
        return f.is_integer()


# common postfixes that consider as a number
postfix_percent = ['%']
postfix_ordinal = ['th', 'st', 'nd', 'rd']
postfix_large_int = ['K', 'M', 'G', 'T', 'P', 'E', 'Z']
postfix_large_int += [f'{x}i' for x in postfix_large_int]
postfixes = postfix_percent + postfix_ordinal + postfix_large_int


def is_number(x: Any, allow_postfix: bool = True) -> bool:
    # intended for proper log styling
    if is_int(x):
        return True
    if is_float(x):
        return True
    if allow_postfix and isinstance(x, str) and len(x) > 0:
        for post in postfixes:
            if x.endswith(post):
                return is_number(x[:-len(post)])
    return False


def clean_whitespace(s: str):
    if not hasattr(clean_whitespace, 'pattern_space'):
        clean_whitespace.pattern_space = re.compile(r'\s+')
    return clean_whitespace.pattern_space.sub(' ', s).strip()


def get_substr_indices(s: str, s_sub: str) -> List[int]:
    s_sub = re.escape(s_sub)
    return [m.start() for m in re.finditer(s_sub, s)]


if __name__ == '__main__':
    def check_int():
        print(is_int(1))
        print(is_int(1.0))
        print(is_int(1.1))
        print(is_int('1'))
        print(is_int('1.0'))
        print(is_int('1.1'))
        print(is_int('1.1', allow_str=True))
        print(is_int('1.0', allow_str=True))
        print(is_int('1', allow_str=True))
        print(is_int('1.1', allow_str=False))
        print(is_int('1.0', allow_str=False))
        print(is_int('1', allow_str=False))
    # check_int()

    def check_num_postfix():
        print(is_number('1'))
        print(is_number('1.0'))
        print(is_number('1.1%'))
        print(is_number('1.1K'))
        print(is_number('1th'))
        print(is_number('1.1th'))
        print(is_number('1.1Mi'))
        print(is_number('M'))
    # check_num_postfix()

    def check_indent():
        text = """### Store Name: ABC Stores (north)

#### Type

retail

#### Room

[room-145]

#### Categories

Specialty Gifts, Specialty Foods

#### Description

Just another chain of convenience stores? Not really! With friendly store people, great service and large selection of items, we make it easy and convenient to get your favorite treasures, flavors and fragrances of the islands.

#### Opening hours

Mo,Tu,We,Th 9:00-23:00; Fr,Sa 9:00-24:00; Su 9:00-23:00

#### Phone

[phone-702-733-7182]"""
        print(indent_str(text))
    check_indent()
