"""
prettier & prettier logging
"""


import re
import math
import string
import datetime
from typing import Tuple, Union


__all__ = [
    'fmt_num', 'fmt_sizeof', 'fmt_delta', 'sec2mmss', 'round_up_1digit', 'nth_sig_digit', 'ordinal', 'round_f', 'fmt_e', 'to_percent',
    'enclose_in_quote',
    'set_pd_style',

    'str2ascii_str', 'sanitize_str',
    'hex2rgb',
    'Timer',
]


def set_pd_style():
    import pandas as pd  # lazy import to save time
    pd.set_option('expand_frame_repr', False)
    pd.set_option('display.precision', 2)
    pd.set_option('max_colwidth', 40)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.min_rows', 16)


def fmt_num(num: Union[float, int], suffix: str = '') -> str:
    """
    Convert number to human-readable format, in e.g. Thousands, Millions
    """
    for unit in ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z']:
        if abs(num) < 1000.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1000.0
    return "%.1f%s%s" % (num, 'Y', suffix)


def fmt_sizeof(num: int, suffix='B', stop_power: Union[int, float] = 1) -> str:
    """ Converts byte size to human-readable format """
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0 ** stop_power:
            n_digit_before_decimal = round(3 * stop_power)
            fmt = f"%{n_digit_before_decimal}.1f%s%s"
            return fmt % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


def fmt_delta(secs: Union[int, float, datetime.timedelta]) -> str:
    if isinstance(secs, datetime.timedelta):
        secs = 86400 * secs.days + secs.seconds + (secs.microseconds/1e6)
    if secs >= 86400:
        d = secs // 86400  # // floor division
        return f'{round(d)}d{fmt_delta(secs - d * 86400)}'
    elif secs >= 3600:
        h = secs // 3600
        return f'{round(h)}h{fmt_delta(secs - h * 3600)}'
    elif secs >= 60:
        m = secs // 60
        return f'{round(m)}m{fmt_delta(secs - m * 60)}'
    else:
        return f'{round(secs)}s'


def sec2mmss(sec: int) -> str:
    return str(datetime.timedelta(seconds=sec))[2:]


def round_up_1digit(num: int):
    d = math.floor(math.log10(num))
    fact = 10**d
    return math.ceil(num/fact) * fact


def nth_sig_digit(flt: float, n: int = 1) -> float:
    """
    :return: first n-th significant digit of `sig_d`
    """
    return float('{:.{p}g}'.format(flt, p=n))


def ordinal(n: int) -> str:
    if 11 <= (n % 100) <= 13:
        suffix = 'th'
    else:
        suffix = ['th', 'st', 'nd', 'rd', 'th'][min(n % 10, 4)]
    return str(n) + suffix


def round_f(x, decimal: int = 2):
    assert isinstance(x, float)
    return round(x, decimal)


def fmt_e(x, decimal: int = 3) -> str:
    assert isinstance(x, float)
    return f'{x:.{decimal}e}'


def to_percent(x, decimal: int = 2, append_char: str = '%') -> Union[str, float]:
    ret = round(x * 100, decimal)
    if append_char is not None:
        ret = f'{ret}{append_char}'
    return ret


def enclose_in_quote(txt: str) -> str:
    """
    Enclose a string in quotes
    """
    # handle cases where the sentence itself is double-quoted, or contain double quotes, use single quotes
    quote = "'" if '"' in txt else '"'
    return f'{quote}{txt}{quote}'


def str2ascii_str(text: str) -> str:
    if not hasattr(str2ascii_str, 'printable'):
        str2ascii_str.printable = set(string.printable)
    return ''.join([x for x in text if x in str2ascii_str.printable])


def sanitize_str(text: str) -> str:
    if not hasattr(sanitize_str, 'whitespace_pattern'):
        sanitize_str.whitespace_pattern = re.compile(r'\s+')
    ret = sanitize_str.whitespace_pattern.sub(' ', str2ascii_str(text)).strip()
    if ret == '':
        raise ValueError(f'Empty text after cleaning, was [{text}]')
    return ret


def hex2rgb(hx: str, normalize=False) -> Union[Tuple[int, ...], Tuple[float, ...]]:
    # Modified from https://stackoverflow.com/a/62083599/10732321
    if not hasattr(hex2rgb, 'regex'):
        hex2rgb.regex = re.compile(r'#[a-fA-F\d]{3}(?:[a-fA-F\d]{3})?$')
    m = hex2rgb.regex.match(hx)
    assert m is not None
    if len(hx) <= 4:
        ret = tuple(int(hx[i]*2, 16) for i in range(1, 4))
    else:
        ret = tuple(int(hx[i:i+2], 16) for i in range(1, 7, 2))
    return tuple(i/255 for i in ret) if normalize else ret


class Timer:
    """
    Counts elapsed time and report in a pretty format

    Intended for logging ML train/test progress
    """
    def __init__(self, start: bool = True):
        self.time_start, self.time_end = None, None
        if start:
            self.start()

    def start(self):
        self.time_start = datetime.datetime.now()

    def end(self):
        if self.time_start is None:
            raise ValueError('Counter not started')

        if self.time_end is not None:
            raise ValueError('Counter already ended')
        self.time_end = datetime.datetime.now()
        return fmt_delta(self.time_end - self.time_start)


if __name__ == '__main__':
    from stefutil.prettier.prettier_debug import s, sic

    def check_time_delta():
        import datetime
        now_ = datetime.datetime.now()
        last_day = now_ - datetime.timedelta(days=1, hours=1, minutes=1, seconds=1)
        sic(now_, last_day)
        diff = now_ - last_day
        sic(diff, fmt_delta(diff))
    # check_time_delta()

    def check_float_pad():
        d = dict(ratio=0.95)
        print(s.i(d))
        print(s.i(d, pad_float=False))
        print(s.pa(d))
        print(s.pa(d, pad_float=False))

        sic(s.pa(d, pad_float=False))
    # check_float_pad()

    def check_ordinal():
        sic([ordinal(n) for n in range(1, 32)])
    # check_ordinal()

    def check_sizeof():
        sz = 4124_1231_4442
        sic(fmt_sizeof(sz, stop_power=2))
        sic(fmt_sizeof(sz, stop_power=1.9))
        sic(fmt_sizeof(sz, stop_power=1.5))
        sic(fmt_sizeof(sz, stop_power=1))
    # check_sizeof()

    def check_style_diff_objects():
        # d = dict(a=1, b=3.0, c=None, d=False, e=True, f='hello')
        # print(s.i(d))
        d = dict(g='5', h='4.2', i='world', j='3.7%')
        # print(s.i(d))
        print(s.i(d, quote_str=True, bold=False))
    # check_style_diff_objects()
