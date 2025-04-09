"""
prettier & prettier logging
"""


import re
import math
import string
import datetime
from typing import Tuple, Union


__all__ = [
    'indent_str',
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


def indent_str(s: str, indent: Union[int, str] = 4) -> str:
    if isinstance(indent, int):
        indent = ' ' * indent
    return '\n'.join([f'{indent}{x}' for x in s.split('\n')])


def fmt_num(num: Union[float, int], suffix: str = '', n_digit: int = 1) -> str:
    """
    Convert number to human-readable format, in e.g. Thousands, Millions
    """
    for unit in ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z']:
        if abs(num) < 1000.0:
            # return "%3.1f%s%s" % (num, unit, suffix)
            return f"{num:.{n_digit}f}{unit}{suffix}"
        num /= 1000.0
    # return "%.1f%s%s" % (num, 'Y', suffix)
    return f"{num:.{n_digit}f}Y{suffix}"


def fmt_sizeof(num: int, suffix='B', stop_power: Union[int, float] = 1) -> str:
    """ Converts byte size to human-readable format """
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0 ** stop_power:
            n_digit_before_decimal = round(3 * stop_power)
            fmt = f"%{n_digit_before_decimal}.1f%s%s"
            return fmt % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


def fmt_delta(secs: Union[int, float, datetime.timedelta], n_digit: int = None) -> str:
    """
    Prettier format time for human readability
    """
    if isinstance(secs, datetime.timedelta):
        secs = 86400 * secs.days + secs.seconds + (secs.microseconds/1e6)
    if secs >= 86400:
        return f'{int(secs // 86400)}d{fmt_delta(secs=secs % 86400, n_digit=n_digit)}'
    elif secs >= 3600:
        return f'{int(secs // 3600)}h{fmt_delta(secs % 3600, n_digit=n_digit)}'
    elif secs >= 60:
        return f'{int(secs // 60)}m{fmt_delta(secs % 60, n_digit=n_digit)}'
    else:
        if isinstance(n_digit, int) and n_digit > 0:
            return f'{secs:.{n_digit}f}s'
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

    def end(self, n_digit_delta: int = 1, prettier: Union[bool, datetime.timedelta] = True) -> str:
        if self.time_start is None:
            raise ValueError('Counter not started')

        if self.time_end is not None:
            raise ValueError('Counter already ended')
        self.time_end = datetime.datetime.now()
        delta = self.time_end - self.time_start
        return fmt_delta(delta, n_digit=n_digit_delta) if prettier else delta


if __name__ == '__main__':
    from stefutil.prettier.prettier_debug import style, icecream as sic

    def check_time_delta():
        import datetime
        now_ = datetime.datetime.now()
        last_day = now_ - datetime.timedelta(days=1, hours=1, minutes=1, seconds=1)
        sic(now_, last_day)
        diff = now_ - last_day
        sic(diff, fmt_delta(diff))
    # check_time_delta()

    def check_time_delta_digits():
        sic(fmt_delta(86523.567, n_digit=3))  # Output: "1d0h2m3.567s"
        sic(fmt_delta(3661.123, n_digit=1))  # Output: "1h1m1.1s"

        sic(fmt_delta(59.987, n_digit=2))
        sic(fmt_delta(59.987, n_digit=1))
        sic(fmt_delta(59.987, n_digit=0))  # Output: "60s"
        sic(fmt_delta(59.987))

        n = 1
        sec = 32424.123412
        sic(f'{sec:.{n}f}s')
    # check_time_delta_digits()

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
        print(style(d, quote_str=True, bold=False))
    # check_style_diff_objects()

    def check_timer():
        import time

        t = Timer()
        time.sleep(1.5)
        delta = t.end(prettier=False)
        sic(delta, type(delta))
    # check_timer()

    def check_fmt_num():
        n = 102342345
        sic(fmt_num(n, n_digit=0))
        sic(fmt_num(n, n_digit=1))
        sic(fmt_num(n, n_digit=2))
        sic(fmt_num(n, n_digit=3))
    check_fmt_num()
