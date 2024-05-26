import os
import re
import json
import pprint
from typing import Tuple, List, Dict, Iterable, Union, Any
from dataclasses import dataclass

from icecream import IceCreamDebugger
from rich.console import Console

from stefutil.primitive import is_float, float_is_sci
from stefutil.prettier import enclose_in_quote


__all__ = [
    'MyIceCreamDebugger', 'sic', 'rc', 'rcl',
    'render_nested_ansi_pairs', 'PrettyStyler', 's',
    '_ANSI_BACKEND', '_ANSI_REST_ALL'
]


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


# syntactic sugar
sic = MyIceCreamDebugger()

rc = Console()
rcl = rc.log


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
    # sep = f'{sep}{idt}'
    idt = indent_str * (indent_level - 1)
    post = f'\n{idt}{postfix}'
    return AdjustIndentOutput(prefix=pref, postfix=post, sep=sep)


# support the `colorama` package for terminal ANSI styling as legacy backend
# by default, use `click.style()` for less & composable code
# default_ansi_backend = 'click'
default_ansi_backend = 'rich'
_ANSI_BACKEND = os.environ.get('SU_ANSI_BACKEND', default_ansi_backend)
if _ANSI_BACKEND not in ['click', 'rich', 'colorama']:
    raise ValueError(f'ANSI backend {_ANSI_BACKEND} not recognized')

if _ANSI_BACKEND == 'click':
    import click

    _ANSI_REST_ALL = '\033[0m'  # taken from `click.termui`
elif _ANSI_BACKEND == 'rich':
    import rich.style

    _ANSI_REST_ALL = '\033[0m'
else:
    assert _ANSI_BACKEND == 'colorama'  # legacy
    import colorama


def render_nested_ansi_pairs(text: str = None):
    """
    process naive (ANSI style, reset) pairs to render as the expected nested pair-wise styling

    user need to ensure that
        1> the ANSI codes are paired,
        2> this function should be called once on such a paired string
    """
    pattern_ansi = re.compile(r'\x1b\[[0-9;]*m')
    reset_code = _ANSI_REST_ALL

    # ============ split into segments by ANSI code ============
    segments = pattern_ansi.split(text)
    codes = pattern_ansi.findall(text)
    assert len(segments) == len(codes) + 1  # sanity check

    # ============ sanity check ansi codes are indeed pairs ============
    malformed = False
    if len(codes) % 2 != 0:
        malformed = True
    if sum(code == reset_code for code in codes) != len(codes) // 2:
        malformed = True
    if malformed:
        raise ValueError(f'ANSI codes in text are not paired in {s.i(text)} - Have you called this rendering function already?')

    active_styles = []  # "active" ANSI style stack
    parts, segments = segments[:1], segments[1:]  # for 1st segment not enclosed in ANSI code

    for segment, code in zip(segments, codes):
        if code == reset_code:
            if active_styles:
                active_styles.pop()
        else:
            active_styles.append(code)

        # ============ enclose each segment in (the corresponding ANSI codes, reset) pair ============
        has_style = len(active_styles) > 0
        if has_style:
            parts.extend(active_styles)

        parts.append(segment)

        if has_style:
            parts.append(reset_code)

    if parts[-1] != reset_code:
        parts.append(reset_code)  # for joining w/ other strings after
    return ''.join(parts)


class PrettyStyler:
    """
    My logging w/ color & formatting, and a lot of syntactic sugar
    """
    if _ANSI_BACKEND in ['click', 'rich']:  # `click.style()` already handles various colors, here stores the mapping from my rep
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
        assert _ANSI_BACKEND == 'colorama'
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
    def style(
            x: Union[int, float, bool, str, None] = None, fg: str = None, bg: str = None, bold: bool = None,
            c_time: str = None, pad: int = None, **style_kwargs
    ) -> str:
        """
        main function for styling, optionally prints to console with timestamp
        """
        args: Dict[str, Any] = PrettyStyler._get_default_style(x)
        args.update({
            k: v for k, v in dict(fg=fg, bg=bg, bold=bold, c_time=c_time, pad=pad, **style_kwargs).items()
            if v is not None
        })
        return PrettyStyler._style(x, **args)

    @staticmethod
    def _get_default_style(x: Union[int, float, bool, str, None]):
        # get custom styling by type of object
        d = PrettyStyler.var_type2style
        if any(x is t for t in [None, True, False]):
            ret = d[x]
        else:
            if is_float(x=x):  # handles the case where `x` is a string representation of a float
                tp = float
            elif isinstance(x, str) and len(x) > 0 and x[-1] == '%' and is_float(x[:-1]):
                tp = float
            else:
                tp = type(x)
            ret = d.get(tp, dict())

        return ret.copy()

    @staticmethod
    def _style(
            x: Union[int, float, bool, str, None] = None, fg: str = None, bg: str = None, bold: bool = False,
            pad: Union[int, str] = None, quote_str: bool = False, **style_kwargs
    ) -> str:
        if isinstance(x, str) and not is_float(x) and quote_str:
            x = enclose_in_quote(x)

        if _ANSI_BACKEND in ['click', 'rich']:
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
                return x
            else:
                x = f'{x:>{pad}}' if pad else x

                style_args = dict(fg=fg, bg=bg, bold=bold, **style_kwargs)
                if _ANSI_BACKEND == 'rich':  # `rich` uses `color` and `bgcolor` instead of `fg` and `bg`
                    style_args['color'] = style_args.pop('fg', None)
                    style_args['bgcolor'] = style_args.pop('bg', None)
                    style = rich.style.Style(**style_args)
                    return style.render(text=str(x))  # explicitly convert to str for `False` and `None` styling
                else:  # `click`
                    return click.style(text=x, **style_args)

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
            return f'{fg}{x:>{pad}}{reset}' if pad else f'{fg}{x}{reset}'

    @staticmethod
    def s(text, fg: str = None, bold: bool = False, with_color: bool = True, **style_kwargs) -> str:
        """
        syntactic sugar for return string instead of print
        """
        if not with_color:
            fg, bg, bold = 'none', 'none', False
        return PrettyStyler.style(text, fg=fg, bold=bold, **style_kwargs)

    @staticmethod
    def i(x, indent: Union[int, float, bool, str, None] = None, indent_str: str = ' ' * 4, render_nested_style: bool = False, **kwargs):
        """
        syntactic sugar for logging `info` as string

        :param x: text to log.
        :param indent: maximum indentation level.
            this will be propagated through dict and list only.
        :param indent_str: string for one level of indentation.
        :param render_nested_style: whether to render nested ANSI styles at the end.
            intended when the input passed in already contains local ANSI styles, see `render_nested_ansi_pairs`.
        """
        if render_nested_style:  # as a syntactic sugar; make a recursive call w/ all other params
            ret = PrettyStyler.i(x, indent=indent, indent_str=indent_str, **kwargs)
            return render_nested_ansi_pairs(ret)

        else:
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
            # elif isinstance(x, (int, float)):
            #     sic('in root float', x)
            #     args = PrettyStyler._get_default_style(x)
            #     x = PrettyStyler._float(x, pad=kwargs.get('pad') or kwargs.pop('pad_float', None))
            #     args.update(kwargs)
            #     return PrettyStyler.i(x, **args)
            else:  # base case
                # kwargs_ = dict(fg='b')
                kwargs_ = dict()
                if _ANSI_BACKEND != 'colorama':  # doesn't support `bold`
                    kwargs_['bold'] = True
                kwargs_.update(kwargs)
                # not needed for base case string styling
                for k in ['for_path', 'value_no_color', 'curr_indent', 'indent_end', 'indent_str', 'container_sep_no_newline']:
                    kwargs_.pop(k, None)
                # get pad value, either from `pad_float` or `pad`
                pad = kwargs_.pop('pad_float', None) or kwargs_.pop('pad', None)
                return PrettyStyler.s(x, pad=pad, **kwargs_)

    @staticmethod
    def _num(n: Union[float, int], pad: Union[int, str] = None) -> Union[str, int, float]:
        if float_is_sci(n):
            return str(n).replace('e-0', 'e-').replace('e+0', 'e+')  # remove leading 0
        elif pad:
            sic('in pad', n, pad)
            return f'{n:{pad}}'
        else:
            return str(n)

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
        return f'{pref}{sep.join([str(e) for e in lst])}{post}'

    @staticmethod
    def _list(
            lst: List, sep: str = None, for_path: bool = False, curr_indent: int = None, indent_end: int = None, indent_str: str = '\t',
            container_sep_no_newline: bool = False, **kwargs
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
            args['pref'], args['post'] = out.prefix, out.postfix
            if all(isinstance(e, (list, dict)) for e in lst) and container_sep_no_newline:
                # by default, indentation will move elements to the next line,
                #   for containers that may potentially get indented in newlines, it's not necessary to add newline here
                #       enabling this will save vertical space
                pass
            else:
                args['sep'] = out.sep
        return PrettyStyler._iter(lst, **args)

    @staticmethod
    def _tuple(tpl: Tuple, **kwargs):
        args = dict(with_color=True, for_path=False, pref='(', post=')')
        args.update(kwargs)
        return PrettyStyler._iter(tpl, **args)

    @staticmethod
    def _dict(
            d: Dict = None,
            with_color=True, pad_float: int = None,  # base case args
            key_value_sep: str = ': ', pairs_sep: str = ', ',  # dict specific args
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
                    return PrettyStyler.i(v, with_color=c, pad_float=pad_float, **kwargs)
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


if __name__ == '__main__':
    from rich.traceback import install
    install()

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

    def check_omit_none():
        d = dict(a=1, b=None, c=3)
        print(s.pa(d))
        print(s.pa(d, omit_none_val=False))
        print(s.pa(d, omit_none_val=True))
    # check_omit_none()

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
                indent_str = '\t'
                print(f'indent={s.i(idt)}: {s.i(d, indent=idt, value_no_color=True, indent_str=indent_str)}')
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

    def check_intense_color():
        print(s.s('hello', fg='m'))
        print(s.s('hello', fg='m', bold=True))
        print(s.s('hello', fg='Bm'))
        print(s.s('hello', fg='Bm', bold=True))
    # check_intense_color()

    def check_coloring():
        from stefutil.prettier.prettier_log import get_logger

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

    def check_nested_style():
        def show_single(text_: str = None):
            text_ansi = render_nested_ansi_pairs(text_)
            print(f'before: {text_}\nafter:  {text_ansi}\n')
            # sic(text_ansi)

        text = s.i(f'hello {s.i("world", fg="y")}! bro')
        show_single(text)

        text = s.i(f'hello {s.i("world", fg="y", italic=True)}! bro')
        show_single(text)

        text = f'say {s.i("hello", italic=True, fg="y")} big {s.i("world", fg="m")}!'
        text = s.i(text, fg="r", bold=False)
        show_single(text)

        text = f'[{text}]'
        show_single(text)

        text = s.i(text, underline=True)
        text = s.i(f'yo {text} hey', fg='b', dim=True)
        text = f'aa {text} bb'
        show_single(text)

        d = dict(a=1, b=2, c=dict(text=text, d='guy'))
        print(d)
        print(s.i(d))
        print(render_nested_ansi_pairs(s.i(d)))
    # check_nested_style()

    def check_ansi_reset():
        # txt = s.i('hello', italic=True, fg="y") + 'world'
        txt = s.i('hello', underline=True, fg="y") + 'world'
        print(txt)
        # sic(txt)
    # check_ansi_reset()

    def check_indent_save_sep_space():
        args = dict()
        indent_str = '\t'
        for lst in [
            [dict(hello=1, a=True), dict(world=2, b=None)],
            [['hello', 'world'], [42, 7]]
        ]:
            for save in [False, True]:
                args['container_sep_no_newline'] = save
                for idt in [1, True]:
                    args['indent'] = idt
                    args_desc = f'[{s.i(args)}]'
                    print(f'{args_desc}: {s.i(lst, indent_str=indent_str, **args)}')
    # check_indent_save_sep_space()

    def check_pad_float():
        mm = 9
        print(s.i(mm, fg='c', pad='02'))
        d = '02'
        print(f'{mm:{d}}')
        rate = 1.23456789
        print(s.i(rate, fg='c', pad='5.2f'))
    check_pad_float()
