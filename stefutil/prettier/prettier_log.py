import os
import re
import sys
import logging
import datetime
from typing import List, Dict, Union, Optional, Any, Callable, Iterable
from collections import OrderedDict

from stefutil.prettier.prettier import hex2rgb
from stefutil.prettier.prettier_debug import style, _DEFAULT_ANSI_BACKEND, _ANSI_REST_ALL


__all__ = [
    'MyTheme', 'MyFormatter',
    'filter_ansi', 'CleanAnsiFileHandler', 'AnsiFileMap',
    'LOG_STR2LOG_LEVEL', 'get_logging_handler', 'get_logger',
    'add_log_handler', 'add_file_handler', 'drop_file_handler', 'set_logger_handler_levels',

    'CheckArg', 'check_arg',

    'now', 'date',
]


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
            nums = [style(num, fg=fg) for num in re.split(r'[\s\-:._]', ret)]
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


def _color_code_string(prompt: str = None) -> str:
    """
    Color-code a prompt for semantic segmentation by a simple heuristic
    """
    # first, split up the prompt into sections
    ret = ''
    sep = '\n\n'
    segs = prompt.split(sep)
    it = iter(segs)
    prev = None
    curr = next(it)
    assert curr is not None
    cs = ['b', 'm', 'r', 'y', 'g']
    i_c = None
    while curr is not None:
        # iteratively check, does the difference between current and next segment indicate a new section?
        # if prev is not None:
        # declare different is current segment is pretty long, via either char count or line count
        long_enough = False
        n_lines = curr.count('\n')
        n_chars = len(curr)
        if n_chars > 250:
            long_enough = True
        elif n_chars > 150 and n_lines > 0:
            long_enough = True
        elif n_lines > 0 and all(len(c) > 60 for c in curr.split('\n')):
            long_enough = True
        elif '\n' not in curr and n_chars > 120:
            long_enough = True
        elif n_lines > 3:
            long_enough = True
        elif '---' in curr or 'Examples:' in curr:
            long_enough = True
        if prev is None:
            i_c = 0
        elif long_enough:
            i_c = (i_c + 1) % len(cs)
        ret += f'{style(curr, fg=cs[i_c])}{sep}'

        prev = curr
        curr = next(it, None)
    return ret


def print_strings(strings: Union[Callable[[], str], List[str], Iterable[str]], n: int = None) -> List[str]:
    """
    color codes a list of strings with heuristics to separate semantic sections

    :param strings: a list of strings or a callable that returns a string
    :param n: number of strings to print
    """
    if isinstance(strings, list):
        assert n is None
        prompts = strings
        n = len(prompts)
    elif hasattr(strings, '__iter__'):
        prompts = list(strings)
        if n is not None:
            n = min(n, len(prompts))
    else:
        n = n or 5
        prompts = [strings() for _ in range(n)]

        if any(not p for p in prompts):
            prompts = [p for p in prompts if p]  # filter out empty/None prompts
            n = len(prompts)
            assert n > 0
    assert all(isinstance(p, str) for p in prompts)  # sanity check

    # prompts = [f'Prompt {i}:\n{pl.i(p)}' for i, p in enumerate(prompts, start=1)]
    prompts = [f'String {i}:\n{_color_code_string(p)}' for i, p in enumerate(prompts, start=1)]
    if n == 1:
        print(prompts[0])
    else:
        for i in range(n):
            sep = '\n\n\n' if i != n - 1 else ''
            print(f'{prompts[i]}{sep}')
    return prompts


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
        import sty
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
    if _DEFAULT_ANSI_BACKEND in ['click', 'rich']:
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
        assert _DEFAULT_ANSI_BACKEND == 'colorama'
        import sty

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

        if _DEFAULT_ANSI_BACKEND in ['click', 'rich']:
            self.time_style_args = MyFormatter.time.copy()
            self.time_style_args.update(style_time or dict())
            self.sep_style_args = MyFormatter.sep.copy()
            self.sep_style_args.update(style_sep or dict())
            self.ref_style_args = MyFormatter.ref.copy()
            self.ref_style_args.update(style_ref or dict())

            color_time = style(MyFormatter.KW_TIME, **self.time_style_args) + style('|', **self.sep_style_args)
        else:
            assert _DEFAULT_ANSI_BACKEND == 'colorama'
            if style_time:
                raise NotImplementedError('Styling for time not supported for `colorama` backend')
            reset = MyFormatter.RESET
            c_time, c_sep = MyFormatter.green, MyFormatter.blue
            color_time = f'{c_time}{MyFormatter.KW_TIME}{c_sep}|{reset}'

        def args2fmt(args_):
            if self.with_color:
                if _DEFAULT_ANSI_BACKEND in ['click', 'rich']:
                    return color_time + self.fmt_meta(*args_) + style(': ', **self.sep_style_args) + MyFormatter.KW_MSG + _ANSI_REST_ALL
                else:
                    assert _DEFAULT_ANSI_BACKEND == 'colorama'
                    return color_time + self.fmt_meta(*args_) + f'{c_sep}: {reset}{MyFormatter.KW_MSG}' + reset
            else:
                return f'{MyFormatter.KW_TIME}|{self.fmt_meta(*args_)}: {MyFormatter.KW_MSG}'

        self.formats = {level: args2fmt(args) for level, args in MyFormatter.LVL_MAP.items()}
        self.formatter = {
            lv: logging.Formatter(fmt, datefmt='%Y-%m-%d %H:%M:%S') for lv, fmt in self.formats.items()
        }

    def fmt_meta(self, meta_abv, meta_style: Union[str, Dict[str, Any]] = None):
        if self.with_color:
            if _DEFAULT_ANSI_BACKEND in ['click', 'rich']:
                return '[' + style(MyFormatter.KW_NAME, **self.ref_style_args) + ']' \
                    + style('::', **self.sep_style_args) + style(MyFormatter.KW_FUNC_NM, **self.ref_style_args) \
                    + style('::', **self.sep_style_args) + style(MyFormatter.KW_FNM, **self.ref_style_args) \
                    + style(':', **self.sep_style_args) + style(MyFormatter.KW_LINENO, **self.ref_style_args) \
                    + style(':', **self.sep_style_args) + style(meta_abv, **meta_style)
            else:
                assert _DEFAULT_ANSI_BACKEND == 'colorama'
                return (f'[{MyFormatter.purple}{MyFormatter.KW_NAME}{MyFormatter.RESET}]'
                        f'{MyFormatter.blue}::{MyFormatter.purple}{MyFormatter.KW_FUNC_NM}'
                        f'{MyFormatter.blue}::{MyFormatter.purple}{MyFormatter.KW_FNM}'
                        f'{MyFormatter.blue}:{MyFormatter.purple}{MyFormatter.KW_LINENO}'
                        f'{MyFormatter.blue}:{meta_style}{meta_abv}{MyFormatter.RESET}')
        else:
            return f'[{MyFormatter.KW_NAME}] {MyFormatter.KW_FUNC_NM}::{MyFormatter.KW_FNM}' \
                   f':{MyFormatter.KW_LINENO}:{meta_abv}'

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


def filter_ansi(txt: str) -> str:
    """
    Removes ANSI escape sequences from the string
    """
    return _ansi_escape.sub('', txt)


class CleanAnsiFileHandler(logging.FileHandler):
    """
    Removes ANSI escape sequences from log file as they are not supported by most text editors
    """
    def emit(self, record):
        record.msg = filter_ansi(record.msg)
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


LogLevel = Union[str, int]
Kind2LogLevel = Dict[str, LogLevel]
LogLevels = Union[LogLevel, Kind2LogLevel]


def _level2int_level(level: LogLevel) -> int:
    if isinstance(level, str):
        return LOG_STR2LOG_LEVEL[level.lower()]
    else:
        assert isinstance(level, int)
        return level


def set_logger_handler_level(logger_or_handler: Union[logging.Logger, logging.Handler] = None, level: LogLevel = None):
    """
    Set logging level for the logger
    """
    if isinstance(level, str):
        level = LOG_STR2LOG_LEVEL[level.lower()]
    else:
        assert isinstance(level, int)
    logger_or_handler.setLevel(level)


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


_file_logging_kinds = ['file', 'colored-file', 'file+colored-file']
_mult_logging_kinds = ['std+file', 'std+file+colored-file', 'file+colored-file']
_logging_kinds = ['stdout'] + _file_logging_kinds + _mult_logging_kinds
_logging_kinds = list(dict.fromkeys(_logging_kinds))  # remove duplicates


def get_logging_handler(
        kind: str = 'stdout', file_path: str = None, level: LogLevels = 'debug', file_mode: str = 'a',
        ansi_file_map: Callable[[str], str] = AnsiFileMap.append_ext
) -> Union[logging.Handler, List[logging.Handler]]:
    """
    :param kind: Handler kind, one of [`stdout`, `file`, `colored-file`, `std+file`, `std+file+colored-file`, `file+colored-file`].
        If `stdout`, handler for stdout
        If `file`, handler for file write (with ANSI style filtering)
        If `colored-file`, handler for file write as is (i.e., without ANSI style filtering)
        If `std+file`, both stdout and file write handlers
        If `std+file+colored-file`, `both` + file write handlers with ANSI style filtering
        If `file+colored-file`, both file write handlers w/ and w/o ANSI style filtering
    :param file_path: File path for file logging.
    :param level: Logging level for the handler.
    :param file_mode: File mode for file logging.
    :param ansi_file_map: Mapping function for the ANSI file handler:
        Returns the mapped file path for ANSI given the original file path.
    """
    if kind in _mult_logging_kinds:  # recursive case
        std, fl_ansi = None, None
        fl = get_logging_handler(kind='file', file_path=file_path, level=level, file_mode=file_mode)

        if 'std' in kind:
            std = get_logging_handler(kind='stdout', level=level)
        if 'colored-file' in kind:
            map_ = ansi_file_map or AnsiFileMap.append_ext
            fl_ansi = get_logging_handler(kind='colored-file', file_path=map_(file_path), level=level, file_mode=file_mode)

        if kind == 'std+file':
            return [std, fl]
        elif kind == 'std+file+colored-file':
            return [std, fl_ansi, fl]
        else:
            assert kind == 'file+colored-file'
            return [fl_ansi, fl]
    else:  # base cases
        if kind == 'stdout':
            handler = logging.StreamHandler(stream=sys.stdout)  # stdout for my own coloring
        else:
            assert kind in ['file', 'colored-file']
            if not file_path:
                raise ValueError(f'{style(file_path)} must be specified for {style("file")} logging')

            dnm = os.path.dirname(file_path)
            if dnm and not os.path.exists(dnm):
                os.makedirs(dnm, exist_ok=True)

            # i.e., when `colored-file`, use the default file handler - no filter out for the ANSI chars
            cls = CleanAnsiFileHandler if kind == 'file' else logging.FileHandler
            handler = cls(file_path, mode=file_mode)
        if isinstance(level, dict):
            level = level['stdout' if kind == 'stdout' else 'file']
        if level:
            set_logger_handler_level(handler, level=level)
        handler.setFormatter(MyFormatter(with_color=kind in ['stdout', 'colored-file']))
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
        logger.info(f'Handlers {style(rmv)} removed')
    return logger


def add_log_handler(logger: logging.Logger = None, level: LogLevels = None, file_path: str = None, kind: str = 'file', drop_prev_handlers: bool = True):
    """
    Adds handler(s) to the logger
    """
    handlers = get_logging_handler(kind=kind, file_path=file_path, level=level)

    if drop_prev_handlers:
        drop_file_handler(logger=logger)

    if not isinstance(handlers, list):
        handlers = [handlers]
    for handler in handlers:
        logger.addHandler(handler)
    return logger


def add_file_handler(logger: logging.Logger = None, level: LogLevels = None, file_path: str = None, kind: str = 'file', drop_prev_handlers: bool = True):
    assert kind in _file_logging_kinds, f'Handler kind {style(kind)} not recognized'
    return add_log_handler(logger=logger, level=level, file_path=file_path, kind=kind, drop_prev_handlers=drop_prev_handlers)


def set_logger_handler_levels(logger: logging.Logger = None, level: LogLevels = None):
    if isinstance(level, dict):
        handler_kd2level = level
    else:
        handler_kd2level = dict(stdout=level, file=level)
    assert set(handler_kd2level.keys()) == {'stdout', 'file'}  # sanity check
    min_level = min(_level2int_level(lvl) for lvl in level.values())
    set_logger_handler_level(logger, level=min_level)

    for handler in logger.handlers:
        # note the condition ordering matters here, for `FileHandler` is a subclass of `StreamHandler`
        if isinstance(handler, logging.FileHandler):
            set_logger_handler_level(logger_or_handler=handler, level=handler_kd2level['file'])

        elif isinstance(handler, logging.StreamHandler):
            set_logger_handler_level(logger_or_handler=handler, level=handler_kd2level['stdout'])

        else:
            raise ValueError(f'Handler {handler} type not recognized')


def get_logger(
        name: str, kind: str = 'stdout', level: Union[LogLevel, Dict[str, LogLevel]] = 'debug', file_path: str = None
) -> logging.Logger:
    """
    :param name: name of the logger.
    :param kind: logger type, one of [`stdout`, `file`, `both`].
        `both` intended for writing to terminal with color and *then* removing styles for file.
    :param level: logging level.
        If dict, expect the corresponding level for each handler kind (one of `stdout`, `file`).
    :param file_path: the file path for file logging.
    """
    assert kind in _logging_kinds, f'Logger kind [{kind}] not recognized'
    logger = logging.getLogger(name)
    logger.handlers = []  # A crude way to remove prior handlers in case of conflict w/ later added handlers

    need_detailed_filtering = False
    if isinstance(level, dict):
        if len(set(level.keys())) > 1:
            need_detailed_filtering = True
        else:  # no need for detailed filtering
            level = level[next(iter(level.keys()))]
    if need_detailed_filtering:
        assert set(level.keys()) == {'stdout', 'file'}  # sanity check
        min_level = min(_level2int_level(lvl) for lvl in level.values())
    else:
        min_level = level
    set_logger_handler_level(logger, level=min_level)

    add_log_handler(logger, level=level if need_detailed_filtering else None, file_path=file_path, kind=kind)
    logger.propagate = False
    return logger


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
                    nm = f'{style(display_name)}::{style(attribute_name)}'
                else:
                    nm = style(display_name)
                CheckArg.logger.warning(f'Argument {nm} is {style("None")} and ignored')
            return True
        if self.verbose:
            d_log = dict(val=val, accepted_values=options)
            CheckArg.logger.info(f'Checking {style(display_name)} w/ {style(d_log)}... ')
        if val not in options:
            if silent:
                return False
            else:
                raise ValueError(f'Unexpected {style(display_name)}: expect one of {style(options)}, got {style(val)}')
        else:
            return True

    def cache_options(self, display_name: str, attr_name: str, options: List[str]):
        if attr_name in self.d_name2func:
            raise ValueError(f'Attribute name {style(attr_name)} already exists')
        self.d_name2func[attr_name] = lambda x: self.assert_options(display_name, x, options, attr_name)
        # set a custom attribute for `attr_name` as the list of options
        setattr(self, attr_name, options)


ca = check_arg = CheckArg()
ca.cache_options(  # See `stefutil::plot.py`
    'Bar Plot Orientation', attr_name='bar_orient', options=['v', 'h', 'vertical', 'horizontal']
)


if __name__ == '__main__':
    from stefutil.prettier.prettier_debug import sic

    # lg = get_logger('test')
    # lg.info('test')

    def check_logger():
        logger = get_logger('blah')
        logger.info('should appear once')
    # check_logger()

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
        logger.info(style(d_log))
        logger.info(style(d_log, indent=True))
        logger.info(style(d_log, indent=True, indent_str=' ' * 4))
        logger.info(style(d_log, indent=True, indent_str='\t'))
        logger.info('only to file', extra=dict(block='stdout'))
    # check_both_handler()

    def check_now():
        sic(now(fmt='full'))
        sic(now(fmt='date'))
        sic(now(fmt='short-date'))
        sic(now(for_path=True, fmt='short-date'))
        sic(now(for_path=True, fmt='date'))
        sic(now(for_path=True, fmt='full'))
        sic(now(for_path=True, fmt='short-full'))
    # check_now()

    def check_color_now():
        print(now(color=True, fmt='short-date'))
        print(now(color=True, for_path=True))
        print(now(color=True))
        print(now(color='g'))
        print(now(color='b'))
    # check_color_now()

    def check_now_tz():
        sic(now())
        sic(now(time_zone='US/Pacific'))
        sic(now(time_zone='US/Eastern'))
        sic(now(time_zone='Europe/London'))
    # check_now_tz()

    def check_date():
        sic(date())
    # check_date()

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

    def check_filter_ansi():
        txt = style('hello')
        print(txt)
        print(filter_ansi(txt))
    # check_filter_ansi()

    def check_print_str():
        lst = [
            "a long string a long string a long string a long string a long string a long string a long string a long string a long string a long string a long string a long string a long string a long string a long string a long string a long string a long string a long string a long string a long string a long string a long string a long string"
            "\n\n\n\n\n\n\n\n\n a long string a long string a long string a long string a long string a long string a long string a long string a long string a long string a long string a long string a long string a long string a long string a long string a long string a long string a long string a long string a long string a long string a long string a long string a long string\n\n\n\n\na long string\n\n\nasdsadasasd",
            'hello world'
        ]
        print_strings(lst)
    # check_print_str()

    def check_diff_log_level():
        path = 'test-diff-log-level.log'
        lg = get_logger('test', kind='std+file+colored-file', level=dict(stdout='warning', file='debug'), file_path=path)
        lg.debug('debug')
        lg.info('info')
        lg.warning('warning')
        lg.error('error')
        lg.critical('critical')

        path = 'test-diff-log-level-start-std.log'
        lg = get_logger('test-start-std')
        add_file_handler(logger=lg, file_path=path, kind='file+colored-file')
        set_logger_handler_levels(logger=lg, level=dict(stdout='INFO', file='DEBUG'))
        lg.debug('debug')
        lg.info('info')
        lg.warning('warning')
        lg.error('error')
        lg.critical('critical')
    check_diff_log_level()
