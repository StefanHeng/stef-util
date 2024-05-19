"""
prettier & prettier logging
"""

import os
import re
import sys
import json
import math
import pprint
import string
import logging
import datetime
from typing import Tuple, List, Dict, Iterable, Union, Optional, Any, Callable
from dataclasses import dataclass
from collections import OrderedDict

from icecream import IceCreamDebugger

from stefutil.primitive import is_float, float_is_sci


__all__ = [
    'fmt_num', 'fmt_sizeof', 'fmt_delta', 'sec2mmss', 'round_up_1digit', 'nth_sig_digit', 'ordinal', 'round_f', 'fmt_e', 'to_percent',
    'set_pd_style',
    'MyIceCreamDebugger', 'sic', 'PrettyStyler', 's',
    'str2ascii_str', 'sanitize_str',
    'hex2rgb', 'MyTheme', 'MyFormatter', 'CleanAnsiFileHandler', 'AnsiFileMap',
    'LOG_STR2LOG_LEVEL', 'get_logging_handler', 'get_logger', 'add_log_handler', 'add_file_handler', 'drop_file_handler',
    'Timer',
    'CheckArg', 'ca',
    'now', 'date'
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


class MyIceCreamDebugger(IceCreamDebugger):
    def __init__(self, output_width: int = 120, **kwargs):
        self._output_width = output_width
        kwargs.update(argToStringFunction=lambda x: pprint.pformat(x, width=output_width))
        super().__init__(**kwargs)
        self.lineWrapWidth = output_width

    @property
    def output_width(self):
        return self._output_width

    @output_width.setter
    def output_width(self, value):
        if value != self._output_width:
            self._output_width = value
            self.lineWrapWidth = value
            self.argToStringFunction = lambda x: pprint.pformat(x, width=value)


sic = MyIceCreamDebugger()


@dataclass
class AdjustIndentOutput:
    prefix: str = None
    postfix: str = None
    sep: str = None


def _adjust_indentation(
        prefix: str = None, postfix: str = None, sep: str = None, indent_level: int = None, indent_str: str = '\t'
) -> AdjustIndentOutput:
    idt = indent_str * indent_level
    pref = f'{prefix}\n{idt}'
    sep = f'{sep.strip()}\n{idt}'
    idt = indent_str * (indent_level - 1)
    post = f'\n{idt}{postfix}'
    return AdjustIndentOutput(prefix=pref, postfix=post, sep=sep)


# support the `colorama` package for terminal ANSI styling as legacy backend
# by default, use `click.style()` for less & composable code
_ansi_backend = os.environ.get('SU_ANSI_BACKEND', 'click')
if _ansi_backend not in ['click', 'colorama']:
    raise ValueError(f'ANSI backend {_ansi_backend} not recognized')
ANSI_BACKEND = _ansi_backend

if ANSI_BACKEND == 'click':
    import click

    _ansi_reset_all = '\033[0m'  # taken from `click
else:
    assert ANSI_BACKEND == 'colorama'  # legacy
    import sty
    import colorama


class PrettyStyler:
    """
    My logging w/ color & formatting, and a lot of syntactic sugar
    """
    if ANSI_BACKEND == 'click':  # `click.style()` already handles various colors, here stores the mapping from my rep
        # start with my shortcut mapping
        short_c2c = dict(
            bl='black',
            r='red',
            g='green',
            y='yellow',
            b='blue',
            m='magenta',
            c='cyan',
            w='white',
        )
        # also add the bright versions, mapping to names used by `click.style()`
        short_c2c.update({f'B{c}': f'bright_{c_}' for c, c_ in short_c2c.items()})
        short_c2c.update(  # now set default colors for each logging type
            log='green',
            warn='yellow',
            error='red',
            success='green',
            info='blue',
            i='blue',
        )

        var_type2style = {
            None: dict(fg='Bm', italic=True),
            True: dict(fg='Bg', italic=True),
            False: dict(fg='Br', italic=True),
            int: dict(fg='Bc'),
            float: dict(fg='Bc'),
            str: dict(fg='Bg'),
        }
    else:
        assert ANSI_BACKEND == 'colorama'
        reset = colorama.Fore.RESET + colorama.Back.RESET + colorama.Style.RESET_ALL
        short_c2c = dict(
            log='',
            warn=colorama.Fore.YELLOW,
            error=colorama.Fore.RED,
            err=colorama.Fore.RED,
            success=colorama.Fore.GREEN,
            suc=colorama.Fore.GREEN,
            info=colorama.Fore.BLUE,
            i=colorama.Fore.BLUE,
            w=colorama.Fore.RED,

            y=colorama.Fore.YELLOW,
            yellow=colorama.Fore.YELLOW,
            red=colorama.Fore.RED,
            r=colorama.Fore.RED,
            green=colorama.Fore.GREEN,
            g=colorama.Fore.GREEN,
            blue=colorama.Fore.BLUE,
            b=colorama.Fore.BLUE,

            m=colorama.Fore.MAGENTA
        )

    @staticmethod
    def log(
            x: Union[int, float, bool, str, None] = None, fg: str = None, bg: str = None, bold: bool = None,
            c_time: str = None, as_str= None, pad: int = None, **style_kwargs
    ) -> str:
        """
        main function for styling, optionally prints to console with timestamp
        """
        args: Dict[str, Any] = PrettyStyler._get_default_style(x)
        args.update({
            k: v for k, v in dict(fg=fg, bg=bg, bold=bold, c_time=c_time, as_str=as_str, pad=pad, **style_kwargs).items()
            if v is not None
        })
        return PrettyStyler._log(x, **args)

    @staticmethod
    def _get_default_style(x: Union[int, float, bool, str, None]):
        # get custom styling by type of object
        d = PrettyStyler.var_type2style
        if any(x is t for t in [None, True, False]):
            ret = d[x]
        else:
            tp = float if is_float(x=x) else type(x)  # handles the case where `x` is a string representation of a float
            ret = d.get(tp, dict())
        return ret.copy()

    @staticmethod
    def _log(
            x: Union[int, float, bool, str, None] = None, fg: str = None, bg: str = None, bold: bool = False,
            c_time='green', as_str=False, pad: int = None, **style_kwargs
    ) -> str:
        if ANSI_BACKEND == 'click':
            if pad:
                raise NotImplementedError
            if fg:
                if fg == 'none':
                    fg = None
                else:
                    fg = PrettyStyler.short_c2c.get(fg, fg)
            if bg:
                if bg == 'none':
                    bg = None
                else:
                    bg = PrettyStyler.short_c2c.get(bg, bg)
            # add the default case when no styling is specified s.t. ANSI reset doesn't contaminate string vars
            if not fg and not bg and not bold and style_kwargs == dict():
                txt = x
            else:
                txt = click.style(text=f'{x:>{pad}}'if pad else x, fg=fg, bg=bg, bold=bold, **style_kwargs)
            if as_str:
                return txt
            else:
                t = click.style(text=now(), fg=c_time, bold=bold)
                print(f'{t}| {txt}')
        else:
            if style_kwargs:
                raise NotImplementedError('Additional styling arguments expected for backend `click` only')

            need_reset = False
            if fg in PrettyStyler.short_c2c:
                fg = PrettyStyler.short_c2c[fg]
                need_reset = True
            if bold:
                fg += colorama.Style.BRIGHT
                need_reset = True
            reset = PrettyStyler.reset if need_reset else ''
            if as_str:
                return f'{fg}{x:>{pad}}{reset}' if pad else f'{fg}{x}{reset}'
            else:
                print(f'{fg}{PrettyStyler.log(now(), fg=c_time, as_str=True)}| {x}{reset}')

    @staticmethod
    def s(text, fg: str = None, bold: bool = False, with_color: bool = True, **style_kwargs) -> str:
        """
        syntactic sugar for return string instead of print
        """
        if not with_color:
            fg, bg, bold = 'none', 'none', False
        return PrettyStyler.log(text, fg=fg, as_str=True, bold=bold, **style_kwargs)

    @staticmethod
    def i(x, indent: Union[int, float, bool, str, None] = None, indent_str: str = ' ' * 4, **kwargs):
        """
        syntactic sugar for logging `info` as string

        :param x: text to log.
        :param indent: maximum indentation level.
            this will be propagated through dict and list only.
        :param indent_str: string for one level of indentation.
        """
        if indent is not None and 'curr_indent' not in kwargs:
            if isinstance(indent, str):
                if indent != 'all':
                    raise ValueError(f'Indentation type {s.i(indent)} not recognized')
                indent = float('inf')
            elif isinstance(indent, bool):
                assert indent is True
                indent = float('inf')
            else:
                assert isinstance(indent, int) and indent > 0  # sanity check
            kwargs['curr_indent'], kwargs['indent_end'] = 1, indent
            kwargs['indent_str'] = indent_str

        # otherwise, already a nested internal call
        if isinstance(x, dict):
            return PrettyStyler._dict(x, **kwargs)
        elif isinstance(x, list):
            return PrettyStyler._list(x, **kwargs)
        elif isinstance(x, tuple):
            return PrettyStyler._tuple(x, **kwargs)
        elif isinstance(x, float):
            args = PrettyStyler._get_default_style(x)
            x = PrettyStyler._float(x, pad=kwargs.get('pad') or kwargs.pop('pad_float', None))
            args.update(kwargs)
            return PrettyStyler.i(x, **args)
        else:  # base case
            # kwargs_ = dict(fg='b')
            kwargs_ = dict()
            if ANSI_BACKEND == 'click':
                kwargs_['bold'] = True
            kwargs_.update(kwargs)
            # not needed for base case string styling
            for k in ['pad_float', 'for_path', 'value_no_color', 'curr_indent', 'indent_end', 'indent_str']:
                kwargs_.pop(k, None)
            return PrettyStyler.s(x, **kwargs_)

    @staticmethod
    def _float(f: float, pad: int = None) -> Union[str, float]:
        if float_is_sci(f):
            return str(f).replace('e-0', 'e-').replace('e+0', 'e+')  # remove leading 0
        elif pad:
            return f'{f:>{pad}}'
        else:
            return str(f)

    @staticmethod
    def pa(text, shorter_bool: bool = True, **kwargs):
        assert isinstance(text, dict)
        fp = 'shorter-bool' if shorter_bool else True
        kwargs = kwargs or dict()
        kwargs['pairs_sep'] = ','  # remove whitespace to save LINUX file path escaping
        return PrettyStyler.i(text, for_path=fp, with_color=False, **kwargs)

    @staticmethod
    def nc(text, **kwargs):
        """
        Syntactic sugar for `i` w/o color
        """
        return PrettyStyler.i(text, with_color=False, **kwargs)

    @staticmethod
    def id(d: Dict) -> str:
        """
        Indented
        """
        return json.dumps(d, indent=4)

    @staticmethod
    def fmt(text) -> str:
        """
        colored by `pygments` & with indent
        """
        from pygments import highlight, lexers, formatters
        return highlight(PrettyStyler.id(text), lexers.JsonLexer(), formatters.TerminalFormatter())

    @staticmethod
    def _iter(
            it: Iterable, with_color=True, pref: str = '[', post: str = ']', sep: str = None, for_path: bool = False,
            curr_indent: int = None, indent_end: int = None, **kwargs
    ):
        # `kwargs` so that customization for other types can be ignored w/o error
        if with_color:
            pref, post = PrettyStyler.s(pref, fg='m'), PrettyStyler.s(post, fg='m')

        def log_elm(e):
            curr_idt = None
            if curr_indent is not None:  # nest indent further down
                assert indent_end is not None  # sanity check
                if curr_indent < indent_end:
                    curr_idt = curr_indent + 1
            if isinstance(e, (list, dict)):
                return PrettyStyler.i(e, with_color=with_color, curr_indent=curr_idt, indent_end=indent_end, for_path=for_path, **kwargs)
            else:
                return PrettyStyler.i(e, with_color=with_color, for_path=for_path, **kwargs)
        lst = [log_elm(e) for e in it]
        if sep is None:
            sep = ',' if for_path else ', '
        return f'{pref}{sep.join(lst)}{post}'

    @staticmethod
    def _list(
            lst: List, sep: str = None, for_path: bool = False, curr_indent: int = None, indent_end: int = None, indent_str: str = '\t',
            **kwargs
    ) -> str:
        args = dict(with_color=True, for_path=False, pref='[', post=']', curr_indent=curr_indent, indent_end=indent_end)
        if sep is None:
            args['sep'] = ',' if for_path else ', '
        else:
            args['sep'] = sep
        args.update(kwargs)

        if curr_indent is not None and len(lst) > 0:
            indent = curr_indent
            pref, post, sep = args['pref'], args['post'], args['sep']
            out = _adjust_indentation(prefix=pref, postfix=post, sep=sep, indent_level=indent, indent_str=indent_str)
            args['pref'], args['post'], args['sep'] = out.prefix, out.postfix, out.sep
        return PrettyStyler._iter(lst, **args)

    @staticmethod
    def _tuple(tpl: Tuple, **kwargs):
        args = dict(with_color=True, for_path=False, pref='(', post=')')
        args.update(kwargs)
        return PrettyStyler._iter(tpl, **args)

    @staticmethod
    def _dict(
            d: Dict = None, with_color=True, pad_float: int = None, key_value_sep: str = ': ', pairs_sep: str = ', ',
            for_path: Union[bool, str] = False, pref: str = '{', post: str = '}',
            omit_none_val: bool = False, curr_indent: int = None, indent_end: int = None, indent_str: str = '\t',
            value_no_color: bool = False, align_keys: Union[bool, int] = False,
            **kwargs
    ) -> str:
        """
        Syntactic sugar for logging dict with coloring for console output
        """
        if align_keys and curr_indent is not None:
            align = 'curr'
            max_c = max(len(k) for k in d.keys()) if len(d) > 0 else None
            if isinstance(align_keys, int) and curr_indent != align_keys:  # check if reaching the level of keys to align
                align = 'pass'
        else:
            align, max_c = None, None

        def _log_val(v):
            curr_idt = None
            need_indent = isinstance(v, (dict, list)) and len(v) > 0
            if need_indent and curr_indent is not None:  # nest indent further down
                assert indent_end is not None  # sanity check
                if curr_indent < indent_end:
                    curr_idt = curr_indent + 1
            c = with_color
            if value_no_color:
                c = False
            if align == 'pass':
                kwargs['align_keys'] = align_keys
            if isinstance(v, dict):
                return PrettyStyler.i(
                    v, with_color=c, pad_float=pad_float, key_value_sep=key_value_sep,
                    pairs_sep=pairs_sep, for_path=for_path, omit_none_val=omit_none_val,
                    curr_indent=curr_idt, indent_end=indent_end, **kwargs
                )
            elif isinstance(v, (list, tuple)):
                return PrettyStyler.i(v, with_color=c, for_path=for_path, curr_indent=curr_idt, indent_end=indent_end, **kwargs)
            else:
                if for_path == 'shorter-bool' and isinstance(v, bool):
                    return 'T' if v else 'F'
                # Pad only normal, expected floats, intended for metric logging
                #   suggest 5 for 2 decimal point percentages
                # elif is_float(v) and pad_float:
                #     if is_float(v, no_int=True, no_sci=True):
                #         v = float(v)
                #         if with_color:
                #             return PrettyLogger.log(v, c='i', as_str=True, pad=pad_float)
                #         else:
                #             return f'{v:>{pad_float}}' if pad_float else v
                #     else:
                #         return PrettyLogger.i(v) if with_color else v
                else:
                    # return PrettyLogger.i(v) if with_color else v
                    return PrettyStyler.i(v, with_color=c, pad_float=pad_float)
        d = d or kwargs or dict()
        if for_path:
            assert not with_color  # sanity check
            key_value_sep = '='
        if with_color:
            key_value_sep = PrettyStyler.s(key_value_sep, fg='m')

        pairs = []
        for k, v_ in d.items():
            if align == 'curr' and max_c is not None:
                k = f'{k:<{max_c}}'
            # no coloring, but still try to make it more compact, e.g. string tuple processing
            k = PrettyStyler.i(k, with_color=False, for_path=for_path)
            if omit_none_val and v_ is None:
                pairs.append(k)
            else:
                pairs.append(f'{k}{key_value_sep}{_log_val(v_)}')
        pairs_sep_ = pairs_sep
        if curr_indent is not None:
            indent = curr_indent
            out = _adjust_indentation(prefix=pref, postfix=post, sep=pairs_sep_, indent_level=indent, indent_str=indent_str)
            pref, post, pairs_sep_ = out.prefix, out.postfix, out.sep
        if with_color:
            pref, post = PrettyStyler.s(pref, fg='m'), PrettyStyler.s(post, fg='m')
        return pref + pairs_sep_.join(pairs) + post


s = PrettyStyler()


def str2ascii_str(text: str) -> str:
    if not hasattr(str2ascii_str, 'printable'):
        str2ascii_str.printable = set(string.printable)
    return ''.join([x for x in text if x in str2ascii_str.printable])


def sanitize_str(text: str) -> str:
    if not hasattr(sanitize_str, 'whitespace_pattern'):
        sanitize_str.whitespace_pattern = re.compile(r'\s+')
    ret = sanitize_str.whitespace_pattern.sub(' ', str2ascii_str(text)).strip()
    if ret == '':
        raise ValueError(f'Empty text after cleaning, was {s.i(text)}')
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


class MyTheme:
    """
    Theme based on `sty` and `Atom OneDark`
    """
    COLORS = OrderedDict([
        ('yellow', 'E5C07B'),
        ('green', '00BA8E'),
        ('blue', '61AFEF'),
        ('cyan', '2AA198'),
        ('red', 'E06C75'),
        ('purple', 'C678DD')
    ])
    yellow, green, blue, cyan, red, purple = (
        hex2rgb(f'#{h}') for h in ['E5C07B', '00BA8E', '61AFEF', '2AA198', 'E06C75', 'C678DD']
    )

    @staticmethod
    def set_color_type(t: str):
        """
        Sets the class attribute accordingly

        :param t: One of [`rgb`, `sty`]
            If `rgb`: 3-tuple of rgb values
            If `sty`: String for terminal styling prefix
        """
        for color, hex_ in MyTheme.COLORS.items():
            val = hex2rgb(f'#{hex_}')  # For `rgb`
            if t == 'sty':
                setattr(sty.fg, color, sty.Style(sty.RgbFg(*val)))
                val = getattr(sty.fg, color)
            setattr(MyTheme, color, val)


class MyFormatter(logging.Formatter):
    """
    Modified from https://stackoverflow.com/a/56944256/10732321

    Default styling: Time in green, metadata indicates severity, plain log message
    """
    if ANSI_BACKEND == 'click':
        # styling for each level and for time prefix
        # time = dict(fg='g')
        # time = dict(fg='Bg', italic=True)
        # time = dict(fg='g', italic=True)
        time = dict(fg='c', italic=True)
        # sep = dict(fg='Bb')  # bright blue
        sep = dict(fg='m')
        # ref = dict(fg='Bm')  # bright magenta
        ref = dict(fg='b')

        debug = dict(fg='none', dim=True, bold=False, italic=True)
        info = dict(fg='none', bold=False, italic=True)
        # info = dict(fg='g')
        warning = dict(fg='y', bold=False, italic=True)
        error = dict(fg='r', bold=False, italic=True)
        critical = dict(fg='m', bold=False, italic=True)
    else:
        assert ANSI_BACKEND == 'colorama'

        RESET = sty.rs.fg + sty.rs.bg + sty.rs.ef

        MyTheme.set_color_type('sty')
        yellow, green, blue, cyan, red, purple = (
            MyTheme.yellow, MyTheme.green, MyTheme.blue, MyTheme.cyan, MyTheme.red, MyTheme.purple
        )

        debug, info, base = RESET
        warning, error, critical = yellow, red, purple
        critical += sty.Style(sty.ef.bold)

    LVL_MAP = {  # level => (abbreviation, style)
        logging.DEBUG: ('DBG', debug),
        logging.INFO: ('INFO', info),
        logging.WARNING: ('WARN', warning),
        logging.ERROR: ('ERR', error),
        logging.CRITICAL: ('CRIT', critical)
    }

    KW_TIME = '%(asctime)s'
    KW_MSG = '%(message)s'
    KW_LINENO = '%(lineno)d'
    KW_FNM = '%(filename)s'
    KW_FUNC_NM = '%(funcName)s'
    KW_NAME = '%(name)s'

    def __init__(
            self, with_color=True, style_time: Dict[str, Any] = None, style_sep: Dict[str, Any] = None, style_ref: Dict[str, Any] = None
    ):
        # time set to green by default, punc separator set to green by default
        super().__init__()
        self.with_color = with_color

        if ANSI_BACKEND == 'click':
            self.time_style_args = MyFormatter.time.copy()
            self.time_style_args.update(style_time or dict())
            self.sep_style_args = MyFormatter.sep.copy()
            self.sep_style_args.update(style_sep or dict())
            self.ref_style_args = MyFormatter.ref.copy()
            self.ref_style_args.update(style_ref or dict())

            color_time = s.s(MyFormatter.KW_TIME, **self.time_style_args) + s.s('|', **self.sep_style_args)
        else:
            assert ANSI_BACKEND == 'colorama'
            if style_time:
                raise NotImplementedError('Styling for time not supported for `colorama` backend')
            reset = MyFormatter.RESET
            c_time, c_sep = MyFormatter.green, MyFormatter.blue
            color_time = f'{c_time}{MyFormatter.KW_TIME}{c_sep}|{reset}'

        def args2fmt(args_):
            if self.with_color:
                if ANSI_BACKEND == 'click':
                    return color_time + self.fmt_meta(*args_) + s.s(': ', **self.sep_style_args) + MyFormatter.KW_MSG + _ansi_reset_all
                else:
                    assert ANSI_BACKEND == 'colorama'
                    return color_time + self.fmt_meta(*args_) + f'{c_sep}: {reset}{MyFormatter.KW_MSG}' + reset
            else:
                return f'{MyFormatter.KW_TIME}|{self.fmt_meta(*args_)}:{MyFormatter.KW_MSG}'

        self.formats = {level: args2fmt(args) for level, args in MyFormatter.LVL_MAP.items()}
        self.formatter = {
            lv: logging.Formatter(fmt, datefmt='%Y-%m-%d %H:%M:%S') for lv, fmt in self.formats.items()
        }

    def fmt_meta(self, meta_abv, meta_style: Union[str, Dict[str, Any]] = None):
        if self.with_color:
            if ANSI_BACKEND == 'click':
                return '[' + s.s(MyFormatter.KW_NAME, **self.ref_style_args) + ']' \
                    + s.s('::', **self.sep_style_args) + s.s(MyFormatter.KW_FUNC_NM, **self.ref_style_args) \
                    + s.s('::', **self.sep_style_args) + s.s(MyFormatter.KW_FNM, **self.ref_style_args) \
                    + s.s(':', **self.sep_style_args) + s.s(MyFormatter.KW_LINENO, **self.ref_style_args) \
                    + s.s(':', **self.sep_style_args) + s.s(meta_abv, **meta_style)
            else:
                assert ANSI_BACKEND == 'colorama'
                return (f'[{MyFormatter.purple}{MyFormatter.KW_NAME}{MyFormatter.RESET}]'
                        f'{MyFormatter.blue}::{MyFormatter.purple}{MyFormatter.KW_FUNC_NM}'
                        f'{MyFormatter.blue}::{MyFormatter.purple}{MyFormatter.KW_FNM}'
                        f'{MyFormatter.blue}:{MyFormatter.purple}{MyFormatter.KW_LINENO}'
                        f'{MyFormatter.blue}:{meta_style}{meta_abv}{MyFormatter.RESET}')
        else:
            return f'[{MyFormatter.KW_NAME}] {MyFormatter.KW_FUNC_NM}::{MyFormatter.KW_FNM}' \
                   f':{MyFormatter.KW_LINENO}, {meta_abv}'

    def format(self, entry):
        return self.formatter[entry.levelno].format(entry)


class HandlerFilter(logging.Filter):
    """
    Blocking messages based on handler
        Intended for sending messages to log file only when both `stdout` and `file` handlers are used
    """
    def __init__(self, handler_name: str = None, **kwargs):
        super().__init__(**kwargs)
        self.handler_name = handler_name

    def filter(self, record: logging.LogRecord) -> bool:
        block = getattr(record, 'block', None)
        if block and self.handler_name == block:
            return False
        else:
            return True


# credit: https://stackoverflow.com/a/14693789/10732321
_ansi_escape = re.compile(r'''
    \x1B  # ESC
    (?:   # 7-bit C1 Fe (except CSI)
        [@-Z\\-_]
    |     # or [ for CSI, followed by a control sequence
        \[
        [0-?]*  # Parameter bytes
        [ -/]*  # Intermediate bytes
        [@-~]   # Final byte
    )
''', re.VERBOSE)


def _filter_ansi(txt: str) -> str:
    """
    Removes ANSI escape sequences from the string
    """
    return _ansi_escape.sub('', txt)


class CleanAnsiFileHandler(logging.FileHandler):
    """
    Removes ANSI escape sequences from log file as they are not supported by most text editors
    """
    def emit(self, record):
        record.msg = _filter_ansi(record.msg)
        super().emit(record)


# taken from HF
LOG_STR2LOG_LEVEL = {
    "detail": logging.DEBUG,  # will also print filename and line number
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


def set_level(logger: Union[logging.Logger, logging.Handler] = None, level: Union[str, int] = None):
    """
    Set logging level for the logger
    """
    if isinstance(level, str):
        level = LOG_STR2LOG_LEVEL[level.lower()]
    logger.setLevel(level)


class AnsiFileMap:
    """
    Some built-in mapping functions for ANSI file handler
    """
    @staticmethod
    def insert_before_log(file_path: str) -> str:
        if file_path.endswith('.log'):
            file_path = file_path[:-4]
        return f'{file_path}.ansi.log'

    @staticmethod
    def append_ext(file_path: str) -> str:
        return f'{file_path}.ansi'


def get_logging_handler(
        kind: str = 'stdout', file_path: str = None, level: Union[str, int] = 'debug',
        ansi_file_map: Callable[[str], str] = AnsiFileMap.append_ext
) -> Union[logging.Handler, List[logging.Handler]]:
    """
    :param kind: Handler kind, one of [`stdout`, `file`, `file-w/-ansi`, `both`, `both+ansi`, `file+ansi`].
        If `stdout`, handler for stdout
        If `file`, handler for file write (with ANSI style filtering)
        If `file-w/-ansi`, handler for file write as is (i.e., without ANSI style filtering)
        If `both`, both stdout and file write handlers
        If `both+ansi`, `both` + file write handlers with ANSI style filtering
        If `file+ansi`, both file write handlers w/ and w/o ANSI style filtering
    :param file_path: File path for file logging.
    :param level: Logging level for the handler.
    :param ansi_file_map: Mapping function for the ANSI file handler:
        Returns the mapped file path for ANSI given the original file path.
    """
    if kind in ['both', 'both+ansi', 'file+ansi']:  # recursive case
        std, fl_ansi = None, None
        fl = get_logging_handler(kind='file', file_path=file_path)

        if kind in ['both', 'both+ansi']:
            std = get_logging_handler(kind='stdout')
        if kind in ['both+ansi', 'file+ansi']:
            map_ = ansi_file_map or AnsiFileMap.append_ext
            fl_ansi = get_logging_handler(kind='file-w/-ansi', file_path=map_(file_path))

        if kind == 'both':
            return [std, fl]
        elif kind == 'both+ansi':
            return [std, fl_ansi, fl]
        else:
            assert kind == 'file+ansi'
            return [fl_ansi, fl]
    else:  # base cases
        if kind == 'stdout':
            handler = logging.StreamHandler(stream=sys.stdout)  # stdout for my own coloring
        else:
            assert kind in ['file', 'file-w/-ansi']
            if not file_path:
                raise ValueError(f'{s.i(file_path)} must be specified for {s.i("file")} logging')

            dnm = os.path.dirname(file_path)
            if dnm and not os.path.exists(dnm):
                os.makedirs(dnm, exist_ok=True)

            # i.e., when `file-w/-ansi`, use the default file handler - no filter out for the ANSI chars
            cls = CleanAnsiFileHandler if kind == 'file' else logging.FileHandler
            handler = cls(file_path)
        set_level(handler, level=level)
        handler.setFormatter(MyFormatter(with_color=kind in ['stdout', 'file-w/-ansi']))
        handler.addFilter(HandlerFilter(handler_name=kind))
        return handler


def drop_file_handler(logger: logging.Logger = None):
    """
    Removes all `FileHandler`s from the logger
    """
    rmv = []
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler):
            logger.removeHandler(h)
            rmv.append(h)
    if len(rmv) > 0:
        logger.info(f'Handlers {s.i(rmv)} removed')
    return logger


def add_log_handler(logger: logging.Logger = None, file_path: str = None, kind: str = 'file', drop_prev_handlers: bool = True):
    """
    Adds handler(s) to the logger
    """
    handlers = get_logging_handler(kind=kind, file_path=file_path)

    if drop_prev_handlers:
        drop_file_handler(logger=logger)

    if not isinstance(handlers, list):
        handlers = [handlers]
    for handler in handlers:
        logger.addHandler(handler)
    return logger


def add_file_handler(logger: logging.Logger = None, file_path: str = None, kind: str = 'file', drop_prev_handlers: bool = True):
    assert kind in ['file', 'file-w/-ansi', 'file+ansi'], f'Handler kind {s.i(kind)} not recognized'
    return add_log_handler(logger, file_path=file_path, kind=kind, drop_prev_handlers=drop_prev_handlers)


def get_logger(
        name: str, kind: str = 'stdout', level: Union[str, int] = 'debug', file_path: str = None
) -> logging.Logger:
    """
    :param name: name of the logger.
    :param kind: logger type, one of [`stdout`, `file`, `both`].
        `both` intended for writing to terminal with color and *then* removing styles for file.
    :param level: logging level.
    :param file_path: the file path for file logging.
    """
    assert kind in ['stdout', 'file-write', 'both', 'both+ansi'], f'Logger kind {s.i(kind)} not recognized'
    logger = logging.getLogger(f'{name} file' if kind == 'file' else name)
    logger.handlers = []  # A crude way to remove prior handlers, ensure only 1 handler per logger
    set_level(logger, level=level)

    add_log_handler(logger, file_path=file_path, kind=kind)
    logger.propagate = False
    return logger


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


class CheckArg:
    """
    An easy, readable interface for checking string arguments as effectively enums

    Intended for high-level arguments instead of actual data processing as not as efficient

    Raise errors when common arguments don't match the expected values
    """
    logger = get_logger('Arg Checker')

    def __init__(self, ignore_none: bool = True, verbose: bool = False):
        """
        :param ignore_none: If true, arguments passed in as `None` will not raise error
        :param verbose: If true, logging messages are print to console
        """
        self.d_name2func = dict()
        self.ignore_none = ignore_none
        self.verbose = verbose

    def __call__(self, **kwargs):
        for k, v in kwargs.items():
            self.d_name2func[k](v)

    def assert_options(
            self, display_name: str, val: Optional[str], options: List[str], attribute_name: str = None, silent: bool = False
    ) -> bool:
        if self.ignore_none and val is None:
            if self.verbose:
                if attribute_name:
                    nm = f'{s.i(display_name)}::{s.i(attribute_name)}'
                else:
                    nm = s.i(display_name)
                CheckArg.logger.warning(f'Argument {nm} is {s.i("None")} and ignored')
            return True
        if self.verbose:
            d_log = dict(val=val, accepted_values=options)
            CheckArg.logger.info(f'Checking {s.i(display_name)} w/ {s.i(d_log)}... ')
        if val not in options:
            if silent:
                return False
            else:
                raise ValueError(f'Unexpected {s.i(display_name)}: expect one of {s.i(options)}, got {s.i(val)}')
        else:
            return True

    def cache_options(self, display_name: str, attr_name: str, options: List[str]):
        if attr_name in self.d_name2func:
            raise ValueError(f'Attribute name {s.i(attr_name)} already exists')
        self.d_name2func[attr_name] = lambda x: self.assert_options(display_name, x, options, attr_name)
        # set a custom attribute for `attr_name` as the list of options
        setattr(self, attr_name, options)


ca = CheckArg()
ca.cache_options(  # See `stefutil::plot.py`
    'Bar Plot Orientation', attr_name='bar_orient', options=['v', 'h', 'vertical', 'horizontal']
)


def now(
        as_str=True, for_path=False, fmt: str = 'short-full', color: Union[bool, str] = False, time_zone: str = None
) -> Union[datetime.datetime, str]:
    """
    # Considering file output path
    :param as_str: If true, returns string; otherwise, returns datetime object
    :param for_path: If true, the string returned is formatted as intended for file system path
        relevant only when as_str is True
    :param color: If true, the string returned is colored
        Intended for terminal logging
        If a string is passed in, the color is applied to the string following `PrettyLogger` convention
    :param fmt: One of [`full`, `date`, `short-date`]
        relevant only when as_str is True
    :param time_zone: Time zone to convert the time to
    """
    d = datetime.datetime.now()

    if time_zone:
        import pytz
        tz = pytz.timezone(time_zone)
        d = d.astimezone(tz)

    if as_str:
        ca.assert_options('Date Format', fmt, ['full', 'short-full', 'date', 'short-date'])
        if 'full' in fmt:
            fmt_tm = '%Y-%m-%d_%H-%M-%S' if for_path else '%Y-%m-%d %H:%M:%S.%f'
        else:
            fmt_tm = '%Y-%m-%d'
        ret = d.strftime(fmt_tm)

        if 'short' in fmt:  # year in 2-digits
            ret = ret[2:]

        if color:
            # split the string on separation chars and join w/ the colored numbers
            fg = color if isinstance(color, str) else 'green'
            nums = [s.s(num, fg=fg) for num in re.split(r'[\s\-:._]', ret)]
            puncs = re.findall(r'[\s\-:._]', ret)
            assert len(nums) == len(puncs) + 1
            ret = ''.join([n + p for n, p in zip(nums, puncs)]) + nums[-1]
            return ret
        return ret
    else:
        return d


def date():
    """
    syntactic sugar for `now()` to just get the date
    """
    return now(for_path=True, fmt='short-date')


if __name__ == '__main__':
    # lg = get_logger('test')
    # lg.info('test')

    def check_log_lst():
        lst = ['sda', 'asd']
        print(s.i(lst))
        # with open('test-logi.txt', 'w') as f:
        #     f.write(pl.nc(lst))
    # check_log_lst()

    def check_log_tup():
        tup = ('sda', 'asd')
        print(s.i(tup))
    # check_log_tup()

    def check_logi():
        d = dict(a=1, b=2)
        txt = 'hello'
        print(s.i(d))
        print(s.i(txt))
        print(s.i(txt, indent=True))
    # check_logi()

    def check_nested_log_dict():
        d = dict(a=1, b=2, c=dict(d=3, e=4, f=['as', 'as']))
        sic(d)
        print(s.i(d))
        print(s.nc(d))
        sic(s.i(d), s.nc(d))
    # check_nested_log_dict()

    def check_logger():
        logger = get_logger('blah')
        logger.info('should appear once')
    # check_logger()

    def check_now():
        sic(now(fmt='full'))
        sic(now(fmt='date'))
        sic(now(fmt='short-date'))
        sic(now(for_path=True, fmt='short-date'))
        sic(now(for_path=True, fmt='date'))
        sic(now(for_path=True, fmt='full'))
        sic(now(for_path=True, fmt='short-full'))
    # check_now()

    def check_ca():
        ori = 'v'
        ca(bar_orient=ori)
    # check_ca()

    def check_ca_warn():
        ca_ = CheckArg(verbose=True)
        ca_.cache_options(display_name='Disp Test', attr_name='test', options=['a', 'b'])
        ca_(test='a')
        ca_(test=None)
        ca_.assert_options('Blah', None, ['hah', 'does not matter'])
    # check_ca_warn()

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

    def check_color_now():
        print(now(color=True, fmt='short-date'))
        print(now(color=True, for_path=True))
        print(now(color=True))
        print(now(color='g'))
        print(now(color='b'))
    # check_color_now()

    def check_omit_none():
        d = dict(a=1, b=None, c=3)
        print(s.pa(d))
        print(s.pa(d, omit_none_val=False))
        print(s.pa(d, omit_none_val=True))
    # check_omit_none()

    def check_both_handler():
        # sic('now creating handler')
        print('now creating handler')

        log_nm, fnm = 'test-both', 'test-both-handler.log'

        # direct = True
        direct = False
        if direct:
            # kd = 'both'
            kd = 'both+ansi'
            logger = get_logger(log_nm, kind=kd, file_path=fnm)
        else:
            logger = get_logger(log_nm, kind='stdout')
            # kd = 'file'
            kd = 'file+ansi'
            add_file_handler(logger, file_path=fnm, kind=kd)

        d_log = dict(a=1, b=2, c='test')
        logger.info(s.i(d_log))
        logger.info(s.i(d_log, indent=True))
        logger.info(s.i(d_log, indent=True, indent_str=' ' * 4))
        logger.info(s.i(d_log, indent=True, indent_str='\t'))
        logger.info('only to file', extra=dict(block='stdout'))
    # check_both_handler()

    def check_pa():
        d = dict(a=1, b=True, c='hell', d=dict(e=1, f=True, g='hell'), e=['a', 'b', 'c'])
        sic(s.pa(d))
        sic(s.pa(d, ))
        sic(s.pa(d, shorter_bool=False))
    # check_pa()

    def check_log_i():
        # d = dict(a=1, b=True, c='hell')
        d = ['asd', 'hel', 'sada']
        print(s.i(d))
        print(s.i(d, with_color=False))
    # check_log_i()

    def check_log_i_float_pad():
        d = {'location': 90.6, 'miscellaneous': 35.0, 'organization': 54.2, 'person': 58.7}
        sic(d)
        print(s.i(d))
        print(s.i(d, pad_float=False))
    # check_log_i_float_pad()

    def check_sci():
        num = 3e-5
        f1 = 84.7
        sic(num, str(num))
        d = dict(md='bla', num=num, f1=f1)
        sic(s.pa(d))
        print(s.i(d))
        print(s.i(num))
    # check_sci()

    def check_pl_iter_sep():
        lst = ['hello', 'world']
        tup = tuple(lst)
        print(s.i(lst, sep='; '))
        print(s.i(tup, sep='; '))
    # check_pl_iter_sep()

    def check_pl_indent():
        ds = [
            dict(a=1, b=dict(c=2, d=3, e=dict(f=1)), c=dict()),
            dict(a=1, b=[1, 2, 3]),
            [dict(a=1, b=2), dict(c=3, d=4)],
            [[1, 2, 3], [4, 5, 6], []]
        ]
        for d in ds:
            for idt in [1, 2, 'all']:
                print(f'indent={s.i(idt)}: {s.i(d, indent=idt, value_no_color=True)}')
    # check_pl_indent()

    def check_pl_color():
        elm = s.i('blah', c='y')
        txt = f'haha {elm} a'
        print(txt)
        s_b = s.s(txt, fg='b')
        print(s_b)
        d = dict(a=1, b=txt)
        print(s.i(d))
        print(s.i(d, value_no_color=True))
    # check_pl_color()

    def check_pl_sep():
        lst = ['haha', '=>']
        print(s.i(lst, sep=' ', pref='', post=''))
    # check_pl_sep()

    def check_align_d():
        d = dict(a=1, bbbbbbbbbb=2, ccccc=dict(d=3, e=4, f=['as', 'as']))
        print(s.i(d))
        print(s.i(d, indent=2))
        print(s.i(d, align_keys=True))
        print(s.i(d, indent=2, align_keys=True))
    # check_align_d()

    def check_align_edge():
        d1 = dict(a=1, bb=2, ccc=dict(d=3, ee=4, fff=['as', 'as']))
        d2 = dict()
        d3 = dict(a=dict())
        for d, aln in [
            (d1, 1),
            (d1, 2),
            (d2, True),
            (d3, True),
            (d3, 2)
        ]:
            print(s.i(d, align_keys=aln, indent=True))
    # check_align_edge()

    def check_dict_tup_key():
        d = {(1, 2): 3, ('foo', 'bar'): 4}
        print(s.i(d))
        d = dict(a=1, b=2)
        print(s.i(d))
    # check_dict_tup_key()

    def check_now_tz():
        sic(now())
        sic(now(time_zone='US/Pacific'))
        sic(now(time_zone='US/Eastern'))
        sic(now(time_zone='Europe/London'))
    # check_now_tz()

    def check_intense_color():
        print(s.s('hello', fg='m'))
        print(s.s('hello', fg='m', bold=True))
        print(s.s('hello', fg='Bm'))
        print(s.s('hello', fg='Bm', bold=True))
    # check_intense_color()

    def check_coloring():
        for i in range(8):
            pref_normal = f'\033[0;3{i}m'
            pref_intense = f'\033[0;9{i}m'
            print(f'{pref_normal}normal{pref_intense}intense')
            # sic(pref_normal, pref_intense)

        for c in ['bl', 'b', 'r', 'g', 'y', 'm', 'c', 'w']:
            bc = f'B{c}'
            txt = c + s.s(f'normal', fg=c) + ' ' + s.s(f'bold', fg=c, bold=True) + ' ' + s.s(f'bright', fg=bc) + ' ' + s.s(f'bright bold', fg=bc, bold=True)
            print(txt)
            # print(txt.replace(' ', '\n')
            # sic(s.s(f'bright', fg=bc))
        # print(s.i('normal', fg='m') + s.i('intense', fg='m', bold=True))

        logger = get_logger(__name__)
        logger.info('hello')
        logger.warning('world')
        logger.error(f"I'm {s.i('Stefan')}")
    # check_coloring()

    def check_date():
        sic(date())
    # check_date()

    def check_sizeof():
        sz = 4124_1231_4442
        sic(fmt_sizeof(sz, stop_power=2))
        sic(fmt_sizeof(sz, stop_power=1.9))
        sic(fmt_sizeof(sz, stop_power=1.5))
        sic(fmt_sizeof(sz, stop_power=1))
    # check_sizeof()

    def check_rich_log():
        import logging
        from rich.logging import RichHandler

        FORMAT = "%(message)s"
        handler = RichHandler(markup=False, highlighter=False)
        handler.setFormatter(MyFormatter())
        logging.basicConfig(
            level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[handler]
        )

        log = logging.getLogger("rich")
        log.info("Hello, World!")
    # check_rich_log()

    def check_style_diff_objects():
        d = dict(a=1, b=3.0, c=None, d=False, e=True, f='hello')
        print(s.i(d))
        d = dict(g='5', h='4.2', i='world')
        print(s.i(d))
    check_style_diff_objects()
