import os
import re
import json
import pprint
from copy import deepcopy
from typing import Tuple, List, Dict, Iterable, Union, Any
from pathlib import Path
from dataclasses import dataclass

from icecream import IceCreamDebugger
from rich.console import Console

from stefutil.primitive import is_float, float_is_sci, is_number
from stefutil.prettier import enclose_in_quote


__all__ = [
    'MyIceCreamDebugger', 'icecream',
    'rich_console', 'rich_console_log',
    '_DEFAULT_ANSI_BACKEND', '_ANSI_REST_ALL',
    'to_rich_markup',
    'render_nested_ansi_pairs', 'Styler', 'style',
]


class MyIceCreamDebugger(IceCreamDebugger):
    def __init__(self, output_width: int = 120, sort_dicts: bool = False, **kwargs):
        self._output_width = output_width
        self._sort_dicts = sort_dicts
        kwargs.update(argToStringFunction=lambda x: pprint.pformat(x, width=output_width, sort_dicts=sort_dicts))
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
            self.argToStringFunction = lambda x: pprint.pformat(x, width=value, sort_dicts=self.sort_dicts)

    @property
    def sort_dicts(self):
        return self._sort_dicts

    @sort_dicts.setter
    def sort_dicts(self, value):
        value = bool(value)
        if value != self._sort_dicts:
            self._sort_dicts = value
            self.argToStringFunction = lambda x: pprint.pformat(x, width=self.output_width, sort_dicts=value)


# syntactic sugar
sic = icecream = MyIceCreamDebugger()

rich_console = Console()
rich_console_log = rich_console.log


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


def _get_container_max_depth(x: Union[Dict, Dict, Tuple, str, Any]) -> int:
    if isinstance(x, dict):
        return 1 + max(_get_container_max_depth(v) for v in x.values())
    elif isinstance(x, list):
        return 1 + max(_get_container_max_depth(e) for e in x)
    # consider tuple as a static value/non-container
    return 0


# support the `colorama` package for terminal ANSI styling as legacy backend
# by default, use `click.style()` for less & composable code
_DEFAULT_ANSI_BACKEND = os.environ.get('SU_ANSI_BACKEND', 'rich')
_ANSI_REST_ALL = '\033[0m'  # taken from `click.termui`


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
        raise ValueError(f'ANSI codes in text are not paired in {style(text)} - Have you called this rendering function already?')

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


def _rich_markup_enclose_single(x=None, tag: str = None) -> str:
    return f'[{tag}]{x}[/{tag}]'


def to_rich_markup(x=None, fg: str = None, bg: str = None, bold: bool = False, italic: bool = False, underline: bool = False):
    x = str(x)

    color = []
    if fg:
        color.append(fg)
    if bg:
        color.append(f'on {bg}')
    if color:
        x = _rich_markup_enclose_single(x=x, tag=' '.join(color))

    for style, tag in [(bold, 'bold'), (italic, 'i'), (underline, 'u')]:
        if style:
            x = _rich_markup_enclose_single(x=x, tag=tag)
    return x


@dataclass
class AnsiStyler:
    def __call__(
            self, x: str = None, fg: str = None, bg: str = None, bold: bool = False, italic: bool = False, underline: bool = False, **kwargs
    ) -> str:
        raise NotImplementedError


@dataclass
class ColoramaStyler(AnsiStyler):
    import colorama

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

    def __call__(
            self, x: str = None, fg: str = None, bg: str = None, bold: bool = False, italic: bool = False, underline: bool = False, **kwargs
    ) -> str:
        import colorama

        if bg or italic or underline or kwargs != dict():
            raise NotImplementedError('Additional styling arguments not supported')

        need_reset = False
        if fg in self.short_c2c:
            fg = self.short_c2c[fg]
            need_reset = True
        if bold:
            fg += colorama.Style.BRIGHT
            need_reset = True
        reset = self.reset if need_reset else ''
        return f'{fg}{x}{reset}'


@dataclass
class ClickNRichStyler(AnsiStyler):
    backend: str = None

    # `click.style()` already handles various colors, here stores the mapping from my representation
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
    
    def get_color(self, color: str = None):
        return None if color == 'none' else self.short_c2c.get(color, color)

    def __call__(
            self, x: str = None, fg: str = None, bg: str = None, bold: bool = False, italic: bool = False, underline: bool = False, **kwargs
    ) -> str:
        fg, bg = self.get_color(color=fg), self.get_color(color=bg)

        # add the default case when no styling is specified s.t. ANSI reset doesn't contaminate string vars
        if not fg and not bg and not bold and not italic and not underline and kwargs == dict():
            return x

        else:
            style_args = dict(fg=fg, bg=bg, bold=bold, italic=italic, underline=underline, **kwargs)
            if self.backend == 'rich':  # `rich` uses `color` and `bgcolor` instead of `fg` and `bg`
                import rich.style
                style_args['color'] = style_args.pop('fg')
                style_args['bgcolor'] = style_args.pop('bg')
                style_ = rich.style.Style(**style_args)
                return style_.render(text=str(x))  # explicitly convert to str for `False` and `None` styling

            elif self.backend == 'rich-markup':
                if kwargs != dict():
                    raise NotImplementedError('Additional styling arguments not supported')
                # now rich rendering may apply additional styling,
                #   e.g., bold-facing container enclosing prefix & postfix (e.g. `{` and `}`)
                return to_rich_markup(x, **style_args)

            else:  # `click`
                import click
                assert self.backend == 'click'
                return click.style(text=x, **style_args)


Single = Union[int, float, bool, str]
Container = Union[dict, list, tuple]
Indent = Union[int, float, bool, str, Dict[str, int]]


class Styler:
    """
    coloring & formatting for python built-in types, and some syntactic sugars for styling configs
    """
    var_type2style = {
        None: dict(fg='Bm', italic=True),
        True: dict(fg='Bg', italic=True),
        False: dict(fg='Br', italic=True),
        int: dict(fg='Bc'),
        float: dict(fg='Bc'),
        'keyword': dict(fg='Bm'),
        str: dict(fg='Bg'),
        'path': dict(fg='m')
    }

    backend2styler_attr = {
        'colorama': 'colorama_styler',  # legacy
        'click': 'click_styler',
        'rich': 'rich_styler',
        'rich-markup': 'rich_markup_styler'
    }

    # lazy loading for optional backend packages
    @staticmethod
    def colorama_styler():
        if not hasattr(Styler, '_colorama_styler'):
            Styler._colorama_styler = ColoramaStyler()
        return Styler._colorama_styler
    
    @staticmethod
    def click_styler():
        if not hasattr(Styler, '_click_styler'):
            Styler._click_styler = ClickNRichStyler(backend='click')
        return Styler._click_styler
    
    @staticmethod
    def rich_styler():
        if not hasattr(Styler, '_rich_styler'):
            Styler._rich_styler = ClickNRichStyler(backend='rich')
        return Styler._rich_styler
    
    @staticmethod
    def rich_markup_styler():
        if not hasattr(Styler, '_rich_markup_styler'):
            # PrettyStyler._rich_markup_styler = RichMarkupStyler()
            Styler._rich_markup_styler = ClickNRichStyler(backend='rich-markup')
        return Styler._rich_markup_styler

    def __init__(
            self, backend: str = _DEFAULT_ANSI_BACKEND,
            indent: Union[Indent, None] = None, fg: str = None, bg: str = None, bold: bool = None, with_color: bool = None
    ):
        self.backend = backend
        self.indent = indent
        self.fg = fg
        self.bg = bg
        self.bold = bold
        self.with_color = with_color

    def __call__(
            self, x: Union[Single, Container, None],
            indent: Union[Indent, None] = None, indent_str: str = ' ' * 4,
            backend: str = _DEFAULT_ANSI_BACKEND,
            render_nested_style: bool = False,
            fg: str = None, bg: str = None, bold: bool = None, with_color: bool = True,
            c_time: str = None, pad: int = None,
            **kwargs
    ):
        """
        a shortcut entry point for all styling calls
        """
        # ================ use instance defaults if not given ================
        backend = backend if backend is not None else self.backend
        indent = indent if indent is not None else self.indent
        fg = fg if fg is not None else self.fg
        bg = bg if bg is not None else self.bg
        bold = bold if bold is not None else self.bold
        with_color = with_color if with_color is not None else self.with_color

        return self.style_container(
            x, indent=indent, indent_str=indent_str, backend=backend, render_nested_style=render_nested_style,
            fg=fg, bg=bg, bold=bold, with_color=with_color, c_time=c_time, pad=pad, **kwargs
        )

    @staticmethod
    def style_single(
            x: Union[Single, None] = None, fg: str = None, bg: str = None, bold: bool = None, with_color: bool = True,
            c_time: str = None, pad: int = None, **style_kwargs
    ) -> str:
        """
        main function for styling single variables (c.f. containers), optionally prints to console with timestamp
        """
        args: Dict[str, Any] = Styler._get_default_style(x)
        args.update({
            k: v for k, v in dict(fg=fg, bg=bg, bold=bold, c_time=c_time, pad=pad, **style_kwargs).items()
            if v is not None
        })
        if not with_color:
            args['fg'], args['bg'], args['bold'] = 'none', 'none', False
        return Styler._style_single(x, **args)

    @staticmethod
    def _get_default_style(x: Union[Single, None]):
        # get custom styling by type of object
        d = Styler.var_type2style
        if any(x is t for t in [None, True, False]):
            ret = d[x]
        else:
            if is_float(x=x):  # handles the case where `x` is a string representation of a float
                tp = float
            elif is_number(x=x):
                tp = float
            elif isinstance(x, Path) \
                    or (isinstance(x, str) and (os.path.exists(x) or len(x) < 256 and x.count(os.sep) >= 2)):  # heuristics to check for path
                tp = 'path'
            # consider `__XXX__` a special keyword
            elif isinstance(x, str) and x.startswith('__') and x.endswith('__'):
                tp = 'keyword'
            else:
                tp = type(x)
            ret = d.get(tp, dict())

        return ret.copy()

    @staticmethod
    def _style_single(
            x: Union[Single, None] = None, fg: str = None, bg: str = None, bold: bool = False,
            pad: Union[int, str] = None, quote_str: bool = False, backend: str = _DEFAULT_ANSI_BACKEND, **style_kwargs
    ) -> str:
        if isinstance(x, str) and not is_float(x) and quote_str:
            x = enclose_in_quote(x)
        if pad:  # if default padding don't work, convert to string and pad
            try:
                x = f'{x:>{pad}}'
            except TypeError:
                x = f'{str(x):>{pad}}'

        if _DEFAULT_ANSI_BACKEND not in ['click', 'rich', 'colorama', 'rich-markup']:
            raise ValueError(f'ANSI backend {_DEFAULT_ANSI_BACKEND} not recognized')
        styler = getattr(Styler, Styler.backend2styler_attr[backend])()
        return styler(x=x, fg=fg, bg=bg, bold=bold, **style_kwargs)

    @staticmethod
    def style_container(
            x: Union[Single, Container, None], bold: bool = True,
            indent: Union[Indent, None] = None, indent_str: str = ' ' * 4, backend: str = _DEFAULT_ANSI_BACKEND,
            render_nested_style: bool = False, no_indent_if_length_1: bool = True, **kwargs
    ):
        """
        style containers (dict, list, tuple) optionally with indentation
        note elements are by default bold-faced

        :param x: a variable to style
        :param bold: whether to bold-face the container
        :param indent: maximum indentation level.
            this will be propagated through dict and list only.
            If True, indent all nested containers
            If a dict, should specify the default indent via a `__default__` key and the indent for each dictionary key via the key
        :param indent_str: string for one level of indentation.
        :param render_nested_style: whether to render nested ANSI styles at the end.
            intended when the input passed in already contains local ANSI styles, see `render_nested_ansi_pairs`.
        :param no_indent_if_length_1: If true and the container has only 1 element, indentation is skipped
        :param backend: backend package for ANSI styling
        """
        if render_nested_style:  # as a syntactic sugar; make a recursive call w/ all other params
            assert backend != 'rich-markup'  # sanity check, this is not the intended use case
            ret = Styler.style_container(
                x, bold=bold, indent=indent, indent_str=indent_str, backend=backend, no_indent_if_length_1=no_indent_if_length_1, **kwargs
            )
            return render_nested_ansi_pairs(ret)

        else:
            if indent is not None and 'curr_indent' not in kwargs:
                if isinstance(indent, str):
                    if indent != 'all':
                        raise ValueError(f'Indentation type {style(indent)} not recognized')
                    indent = float('inf')
                if isinstance(indent, dict):
                    assert len(indent) >= 2 and '__default__' in indent  # sanity check
                    kwargs['indent_config'] = indent_config = indent
                    indent = indent_config.pop('__default__')
                elif isinstance(indent, bool):
                    assert indent is True
                    indent = float('inf')
                else:
                    assert isinstance(indent, int) and indent != 0  # sanity check

                if indent < 0:  # negative indent for counting indent backwards
                    # => figure out the proper forward indent level
                    indent = _get_container_max_depth(x) + indent

                kwargs['curr_indent'], kwargs['indent_end'] = 1, indent
            kwargs['indent_str'] = indent_str
            kwargs['no_indent_if_length_1'] = no_indent_if_length_1
            kwargs['backend'] = backend
            kwargs['bold'] = bold

            # otherwise, already a nested internal call
            if isinstance(x, dict):
                return Styler._dict(x, **kwargs)
            elif isinstance(x, list):
                return Styler._list(x, **kwargs)
            elif isinstance(x, tuple):
                return Styler._tuple(x, **kwargs)

            else:  # base case for single variables
                kwargs_ = dict()
                if _DEFAULT_ANSI_BACKEND == 'colorama' and bold:
                    raise ValueError(f'`bold` styling not supported for ANSI backend {_DEFAULT_ANSI_BACKEND}')
                kwargs_.update(kwargs)
                # not needed for base case string styling
                for k in [
                    'curr_indent', 'indent_end', 'indent_str', 'no_indent_if_length_1', 'align_keys',
                    'for_path', 'brace_no_color', 'value_no_color', 'color_keys', 'container_sep_no_newline'
                ]:
                    kwargs_.pop(k, None)
                if 'pad_float' in kwargs_:  # resolve API differences
                    assert kwargs_.get('pad', None) is None
                    kwargs_['pad'] = kwargs_.pop('pad_float')
                return Styler.style_single(x, **kwargs_)

    @staticmethod
    def nb(x, **kwargs):  # syntax sugar for style w/o bold
        kwargs['bold'] = False
        return Styler.style_container(x, **kwargs)

    @staticmethod
    def _num(n: Union[float, int], pad: Union[int, str] = None) -> Union[str, int, float]:
        if float_is_sci(n):
            return str(n).replace('e-0', 'e-').replace('e+0', 'e+')  # remove leading 0
        elif pad:
            return f'{n:{pad}}'
        else:
            return str(n)

    @staticmethod
    def pa(text, shorter_bool: bool = True, **kwargs):
        """
        Syntactic sugar for style for file paths
        """
        assert isinstance(text, dict)
        fp = 'shorter-bool' if shorter_bool else True
        kwargs = kwargs or dict()
        kwargs['pairs_sep'] = ','  # remove whitespace to save LINUX file path escaping
        return Styler.style_container(text, for_path=fp, with_color=False, **kwargs)

    @staticmethod
    def nc(text, **kwargs):
        """
        Syntactic sugar for style w/o color
        """
        kwargs['with_color'] = False
        return Styler.style_container(text, **kwargs)

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
        return highlight(Styler.id(text), lexers.JsonLexer(), formatters.TerminalFormatter())

    @staticmethod
    def _iter(
            it: Iterable, with_color=True, pref: str = '[', post: str = ']', sep: str = None, for_path: bool = False,
            curr_indent: int = None, indent_end: int = None, indent_str: str = '\t', no_indent_if_length_1: bool = None,
            brace_no_color: bool = False, backend: str = _DEFAULT_ANSI_BACKEND,
            **kwargs
    ):
        # `kwargs` so that customization for other types can be ignored w/o error
        if with_color and not brace_no_color:
            pref, post = Styler.style_single(pref, fg='m', backend=backend), Styler.style_single(post, fg='m', backend=backend)

        def log_elm(e):
            curr_idt = None
            if curr_indent is not None:  # nest indent further down
                assert indent_end is not None  # sanity check
                if curr_indent < indent_end:
                    curr_idt = curr_indent + 1
            args = dict(with_color=with_color, for_path=for_path, backend=backend, brace_no_color=brace_no_color)
            if isinstance(e, (list, dict)):
                return Styler.style_container(
                    e, curr_indent=curr_idt, indent_end=indent_end, indent_str=indent_str, no_indent_if_length_1=no_indent_if_length_1,
                    **args, **kwargs)
            else:
                return Styler.style_container(e, **args, **kwargs)
        lst = [log_elm(e) for e in it]
        if sep is None:
            sep = ',' if for_path else ', '
        return f'{pref}{sep.join([str(e) for e in lst])}{post}'

    @staticmethod
    def _list(
            lst: List, sep: str = None, for_path: bool = False, curr_indent: int = None, indent_end: int = None, indent_str: str = '\t',
            container_sep_no_newline: bool = False, no_indent_if_length_1: bool = None, **kwargs
    ) -> str:
        args = dict(
            with_color=True, for_path=False, pref='[', post=']',
            curr_indent=curr_indent, indent_end=indent_end, indent_str=indent_str, no_indent_if_length_1=no_indent_if_length_1)
        if sep is None:
            args['sep'] = ',' if for_path else ', '
        else:
            args['sep'] = sep
        args.update(kwargs)

        need_indent = curr_indent is not None and len(lst) > 0
        if need_indent and no_indent_if_length_1 and len(lst) == 1:
            need_indent = None
        if need_indent:
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
        return Styler._iter(lst, **args)

    @staticmethod
    def _tuple(tpl: Tuple, **kwargs):
        args = dict(with_color=True, for_path=False, pref='(', post=')')
        args.update(kwargs)
        return Styler._iter(tpl, **args)

    @staticmethod
    def _dict(
            d: Dict = None,
            with_color=True, pad_float: int = None,  # base case args
            key_value_sep: str = ': ', pairs_sep: str = ', ',  # dict specific args
            for_path: Union[bool, str] = False, pref: str = '{', post: str = '}',
            omit_none_val: bool = False, curr_indent: int = None, indent_end: int = None, indent_str: str = '\t', indent_config: Dict[str, int] = None,
            no_indent_if_length_1: bool = None,
            brace_no_color: bool = False, color_keys: bool = False, value_no_color: bool = False, align_keys: Union[bool, int] = False,
            backend: str = _DEFAULT_ANSI_BACKEND, **kwargs
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

        def _log_val(k, v):
            curr_idt = None
            need_indent = isinstance(v, (dict, list)) and len(v) > 0
            if need_indent and curr_indent is not None:  # nest indent further down
                assert indent_end is not None  # sanity check
                if indent_config is not None:
                    indent_end_ = indent_config.get(k, indent_end)
                else:
                    indent_end_ = indent_end
                if curr_indent < indent_end_:
                    curr_idt = curr_indent + 1
            c = with_color
            if value_no_color:
                c = False
            if align == 'pass':
                kwargs['align_keys'] = align_keys
            if isinstance(v, dict):
                return Styler.style_container(
                    v, with_color=c, pad_float=pad_float, key_value_sep=key_value_sep,
                    pairs_sep=pairs_sep, for_path=for_path, omit_none_val=omit_none_val,
                    curr_indent=curr_idt, indent_end=indent_end, indent_str=indent_str, no_indent_if_length_1=no_indent_if_length_1,
                    backend=backend, brace_no_color=brace_no_color, color_keys=color_keys, value_no_color=value_no_color, **kwargs
                )
            elif isinstance(v, (list, tuple)):
                return Styler.style_container(
                    v, with_color=c, for_path=for_path,
                    curr_indent=curr_idt, indent_end=indent_end, indent_str=indent_str, no_indent_if_length_1=no_indent_if_length_1,
                    backend=backend, brace_no_color=brace_no_color, color_keys=color_keys, value_no_color=value_no_color, **kwargs)
            else:
                if for_path == 'shorter-bool' and isinstance(v, bool):
                    return 'T' if v else 'F'
                else:
                    return Styler.style_container(v, with_color=c, pad_float=pad_float, backend=backend, **kwargs)
        if not d:
            if kwargs:
                d = deepcopy(kwargs)
                for k in ['fg', 'bg', 'c_time', 'pad', 'bold']:  # drop the keyword args for single-value styling
                    d.pop(k, None)
            else:
                d = dict()
        if for_path:
            assert not with_color  # sanity check
            key_value_sep = '='
        if with_color:
            key_value_sep = Styler.style_single(key_value_sep, fg='m', backend=backend)

        pairs = []
        for k, v_ in d.items():
            if align == 'curr' and max_c is not None:
                k = f'{k:<{max_c}}'
            # no coloring by default, but still try to make it more compact, e.g., string tuple processing
            k = Styler.style_container(k, with_color=color_keys, for_path=for_path, backend=backend, brace_no_color=brace_no_color, **kwargs)
            if omit_none_val and v_ is None:
                pairs.append(k)
            else:
                pairs.append(f'{k}{key_value_sep}{_log_val(k, v_)}')
        pairs_sep_ = pairs_sep
        need_indent = curr_indent is not None
        if need_indent and no_indent_if_length_1 and len(pairs) == 1:
            need_indent = False
        if need_indent:
            indent = curr_indent
            out = _adjust_indentation(prefix=pref, postfix=post, sep=pairs_sep_, indent_level=indent, indent_str=indent_str)
            pref, post, pairs_sep_ = out.prefix, out.postfix, out.sep
        if with_color and not brace_no_color:
            pref, post = Styler.style_single(pref, fg='m', backend=backend), Styler.style_single(post, fg='m', backend=backend)
        return pref + pairs_sep_.join(pairs) + post


style = Styler()


if __name__ == '__main__':
    from rich.traceback import install
    install()

    s = style

    def check_log_lst():
        lst = ['sda', 'asd']
        print(style(lst))
        # with open('test-logi.txt', 'w') as f:
        #     f.write(pl.nc(lst))
    # check_log_lst()

    def check_log_tup():
        tup = ('sda', 'asd')
        print(style(tup))
    # check_log_tup()

    def check_logi():
        d = dict(a=1, b=2)
        txt = 'hello'
        print(style(d))
        print(style(txt))
        print(style(txt, indent=True))
    # check_logi()

    def check_nested_log_dict():
        d = dict(a=1, b=2, c=dict(d=3, e=4, f=['as', 'as']))
        sic(d)
        print(style(d))
        print(s.nc(d))
        sic(style(d), s.nc(d))
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
        print(style(d))
        print(style(d, with_color=False))
    # check_log_i()

    def check_log_i_float_pad():
        d = {'location': 90.6, 'miscellaneous': 35.0, 'organization': 54.2, 'person': 58.7}
        sic(d)
        print(style(d))
        print(style(d, pad_float=False))
    # check_log_i_float_pad()

    def check_sci():
        num = 3e-5
        f1 = 84.7
        sic(num, str(num))
        d = dict(md='bla', num=num, f1=f1)
        sic(s.pa(d))
        print(style(d))
        print(style(num))
    # check_sci()

    def check_pl_iter_sep():
        lst = ['hello', 'world']
        tup = tuple(lst)
        print(style(lst, sep='; '))
        print(style(tup, sep='; '))
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
                print(f'indent={style(idt)}: {style(d, indent=idt, value_no_color=True, indent_str=indent_str)}')
    # check_pl_indent()

    def check_pl_color():
        elm = style('blah', c='y')
        txt = f'haha {elm} a'
        print(txt)
        s_b = style(txt, fg='b')
        print(s_b)
        d = dict(a=1, b=txt)
        print(style(d))
        print(style(d, value_no_color=True))
    # check_pl_color()

    def check_pl_sep():
        lst = ['haha', '=>']
        print(style(lst, sep=' ', pref='', post=''))
    # check_pl_sep()

    def check_align_d():
        d = dict(a=1, bbbbbbbbbb=2, ccccc=dict(d=3, e=4, f=['as', 'as']))
        print(style(d))
        print(style(d, indent=2))
        print(style(d, align_keys=True))
        print(style(d, indent=2, align_keys=True))

        d = {
            '#': {
                'Chemical': {
                    '__correct__': 132, '__not_named_entity__': 5, '__wrong_boundary__': 9, '__wrong_type__': 43, 'incorrect': 57},
                'total': {
                    '__correct__': 132, '__not_named_entity__': 5, '__wrong_boundary__': 9, '__wrong_type__': 43, 'incorrect': 57}
            },
            '%': {
                'Chemical': {
                    '__correct__': '69.8%', '__not_named_entity__': '2.6%', '__wrong_boundary__': '4.8%', '__wrong_type__': '22.8%', 'incorrect': '30.2%'},
                'total': {
                    '__correct__': '69.8%', '__not_named_entity__': '2.6%', '__wrong_boundary__': '4.8%', '__wrong_type__': '22.8%', 'incorrect': '30.2%'}
            }
        }
        print(style(d, align_keys=2, indent=2, pad=5))
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
            print(style(d, align_keys=aln, indent=True))
    # check_align_edge()

    def check_dict_tup_key():
        d = {(1, 2): 3, ('foo', 'bar'): 4}
        print(style(d))
        d = dict(a=1, b=2)
        print(style(d))
    # check_dict_tup_key()

    def check_intense_color():
        print(style('hello', fg='m'))
        print(style('hello', fg='m', bold=True))
        print(style('hello', fg='Bm'))
        print(style('hello', fg='Bm', bold=True))
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
            txt = c + style(f'normal', fg=c) + ' ' + style(f'bold', fg=c, bold=True) + ' ' + style(f'bright', fg=bc) + ' ' + style(f'bright bold', fg=bc, bold=True)
            print(txt)
            # print(txt.replace(' ', '\n')
            # sic(style(f'bright', fg=bc))
        # print(s.i('normal', fg='m') + s.i('intense', fg='m', bold=True))

        logger = get_logger(__name__)
        logger.info('hello')
        logger.warning('world')
        logger.error(f"I'm {style('Stefan')}")
    # check_coloring()

    def check_nested_style():
        def show_single(text_: str = None):
            text_ansi = render_nested_ansi_pairs(text_)
            print(f'before: {text_}\nafter:  {text_ansi}\n')
            # sic(text_ansi)

        text = style(f'hello {style("world", fg="y")}! bro')
        show_single(text)

        text = style(f'hello {style("world", fg="y", italic=True)}! bro')
        show_single(text)

        text = f'say {style("hello", italic=True, fg="y")} big {style("world", fg="m")}!'
        text = style(text, fg="r", bold=False)
        show_single(text)

        text = f'[{text}]'
        show_single(text)

        text = style(text, underline=True)
        text = style(f'yo {text} hey', fg='b', dim=True)
        text = f'aa {text} bb'
        show_single(text)

        d = dict(a=1, b=2, c=dict(text=text, d='guy'))
        print(d)
        print(style(d))
        print(render_nested_ansi_pairs(style(d)))
    # check_nested_style()

    def check_ansi_reset():
        # txt = s.i('hello', italic=True, fg="y") + 'world'
        txt = style('hello', underline=True, fg="y") + 'world'
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
                    args_desc = f'[{style(args)}]'
                    print(f'{args_desc}: {style(lst, indent_str=indent_str, **args)}')
    # check_indent_save_sep_space()

    def check_pad_float():
        mm = 9
        print(style(mm, fg='c', pad='02'))
        d = '02'
        print(f'{mm:{d}}')
        rate = 1.23456789
        print(style(rate, fg='c', pad='5.2f'))
    # check_pad_float()

    def check_rich_markup():
        from rich import print as rich_print

        styles = [
            dict(fg='red'), dict(bg='bright_blue'), dict(fg='black', bg='white'),
            dict(bold=True), dict(italic=True, underline=True), dict(fg='cyan', underline=True)
        ]
        for style in styles:
            txt = 'hello world'
            # print(s.i(txt, **style, backend='rich-markup'))
            rich_print(to_rich_markup(txt, **style))

            fg = style.pop('fg', 'none')
            bold = style.pop('bold', False)
            rich_print(style(txt, backend='rich-markup', fg=fg, bold=bold, **style))

        # txt = 'hello world'
        # print(s.i(txt, bold=True, backend='rich-markup'))
        d = dict(hello=1, world=4.2, txt='hey', d=dict(a=1, b=2))
        lst = [1, 4.2, 'hello', ['a', 'b'], dict(a=1, b=2)]

        def print_single(x):
            rich_print('rich markup backend:', style(x, backend='rich-markup', brace_no_color=True))
            print('default rich backend:', style(lst, brace_no_color=True))
            # sic(s.i(x, backend='rich-markup', brace_no_color=True))
        print_single(d)
        print_single(lst)
    # check_rich_markup()

    def check_i_customization():
        d = dict(a=1, b=2, c=dict(d=3, e=4, f=['as', 'but']))
        print(s.nb(d, indent=2))
        d = {'a': 1, 42: (1, 2), 'c': 'hello'}
        print(style(d, color_keys=True, value_no_color=True))
    # check_i_customization()

    def check_style_path():
        path = '/home/stefan/Downloads'
        print(style(path))
        path = Path('/home/stefan/Downloads')
        print(style(path))
    # check_style_path()

    def check_pad():
        """
         {
                random        : {pair-rm: 833, rm-gemma: 605, stable-lm: None},
                disagree-range: {pair-rm: 299, rm-gemma: None, stable-lm: None},
                disagree-min  : {pair-rm: 269, rm-gemma: None, stable-lm: None},
                uncertain     : {pair-rm: 857, rm-gemma: 1041, stable-lm: None}
        }
        """
        d = {
            'random': {'pair-rm': 833, 'rm-gemma': 605, 'stable-lm': None},
            'disagree-range': {'pair-rm': 299, 'rm-gemma': None, 'stable-lm': None},
            'disagree-min': {'pair-rm': 269, 'rm-gemma': None, 'stable-lm': None},
            'uncertain': {'pair-rm': 857, 'rm-gemma': 1041, 'stable-lm': None}
        }
        print(style(d, align_keys=1, indent=1, pad=4))
    # check_pad()

    def check_style_keyword():
        d = {'__correct__': 1, '__not_named_entity__': 2, '__wrong_boundary__': 3, '__wrong_type__': 4, 'incorrect ': 5}
        print(style(d, color_keys=True))
    # check_style_keyword()

    def check_backward_indent():
        d = {'a': {'b': {'c': {'d': {'e': 1}}}}}
        print(s(d, indent=-1))
        print(s(d, indent=-2))
        print(s(d, indent=-3))

        l = [[[[[[1]]]]]]
        print(s(l, indent=-1))
        print(s(l, indent=-2))
        print(s(l, indent=-3))
    # check_backward_indent()

    def check_float_pad():
        d = dict(ratio=0.95)
        print(style(d))
        print(style(d, pad_float=False))
        print(style.pa(d))
        print(style.pa(d, pad_float=False))

        sic(style.pa(d, pad_float=False))
    # check_float_pad()

    def check_log_num():
        d = dict(a='1%', b='3K', c='10th', d='1st', e='hello')
        print(style(d))
    # check_log_num()

    def check_no_indent_if_len_1():
        lst = ['hello', 'world']
        print(style(lst, indent=True))
        lst = ['yo']
        print(style(lst, indent=True))
        dct = dict(a='h')
        print(style(dct, indent=True))

        lst = [dict(a='h')]
        print(style(lst, indent=True))

        # dct = dict(a=[1])
        dct = dict(a=[1], b=[2, 3])
        print(style(dct, indent=True))
    # check_no_indent_if_len_1()

    def check_custom_indent_cfg():
        d = dict(a=[1, 2], b=[3, 4])
        print(style(d, indent=True))
        print(style(d, indent=1))
        print(style(d, indent=dict(__default__=1, b=2)))
    # check_custom_indent_cfg()

    def check_empty():
        print(style(dict()))
        print(style(list()))
        print(style(tuple()))
    check_empty()
