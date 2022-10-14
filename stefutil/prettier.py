"""
prettier & prettier logging
"""

import os
import re
import sys
import json
import math
import pprint
import logging
import datetime
from typing import Tuple, List, Dict, Iterable, Any, Union
from pygments import highlight, lexers, formatters
from collections import OrderedDict
from collections.abc import Sized

import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from transformers import Trainer, TrainerCallback
import sty
import colorama
from tqdm.auto import tqdm
from icecream import IceCreamDebugger

from stefutil.primitive import is_float


__all__ = [
    'fmt_num', 'fmt_sizeof', 'fmt_delta', 'sec2mmss', 'round_up_1digit', 'nth_sig_digit',
    'MyIceCreamDebugger', 'mic',
    'PrettyLogger', 'pl',
    'hex2rgb', 'MyTheme', 'MyFormatter', 'get_logger',
    'CheckArg', 'ca',
    'now',
    'MlPrettier', 'MyProgressCallback', 'LogStep'
]


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


def fmt_sizeof(num: int, suffix='B') -> str:
    """ Converts byte size to human-readable format """
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
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
            self.lineWrapgitWidth = value
            self.argToStringFunction = lambda x: pprint.pformat(x, width=value)


mic = MyIceCreamDebugger()


class PrettyLogger:
    """
    My logging w/ color & formatting, and a lot of syntactic sugar
    """
    reset = colorama.Fore.RESET + colorama.Back.RESET + colorama.Style.RESET_ALL
    key2c = dict(
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
    def log(s, c: str = 'log', c_time='green', as_str=False, bold: bool = False, pad: int = None):
        """
        Prints `s` to console with color `c`
        """
        need_reset = False
        if c in PrettyLogger.key2c:
            c = PrettyLogger.key2c[c]
            need_reset = True
        if bold:
            c += colorama.Style.BRIGHT
            need_reset = True
        reset = PrettyLogger.reset if need_reset else ''
        if as_str:
            return f'{c}{s:>{pad}}{reset}' if pad is not None else f'{c}{s}{reset}'
        else:
            print(f'{c}{PrettyLogger.log(now(), c=c_time, as_str=True)}| {s}{reset}')

    @staticmethod
    def s(s, c: str = None, bold: bool = False, with_color: bool = True) -> str:
        """
        syntactic sugar for return string instead of print
        """
        c = c if with_color else ''  # keeping the same signature with logging specific types for `lognc`
        return PrettyLogger.log(s, c=c, as_str=True, bold=bold)

    @staticmethod
    def i(s, **kwargs):
        """
        Syntactic sugar for logging `info` as string
        """
        if isinstance(s, dict):
            return PrettyLogger._dict(s, **kwargs)
        elif isinstance(s, list):
            return PrettyLogger._list(s, **kwargs)
        elif isinstance(s, tuple):
            return PrettyLogger._tuple(s, **kwargs)
        else:
            kwargs_ = dict(c='i')
            kwargs_.update(kwargs)
            return PrettyLogger.s(s, **kwargs_)

    @staticmethod
    def pa(s):
        assert isinstance(s, dict)
        return PrettyLogger.i(s, for_path=True, with_color=False)

    @staticmethod
    def nc(s, **kwargs):
        """
        Syntactic sugar for `i` w/o color
        """
        return PrettyLogger.i(s, with_color=False, **kwargs)

    @staticmethod
    def id(d: Dict) -> str:
        """
        Indented
        """
        return json.dumps(d, indent=4)

    @staticmethod
    def fmt(s) -> str:
        """
        colored by `pygments` & with indent
        """
        return highlight(PrettyLogger.id(s), lexers.JsonLexer(), formatters.TerminalFormatter())

    @staticmethod
    def _iter(it: Iterable, with_color=True, pref: str = '[', post: str = ']'):
        if with_color:
            pref, post = PrettyLogger.s(pref, c='m'), PrettyLogger.s(post, c='m')
        lst = [PrettyLogger.i(e, with_color=with_color) for e in it]
        return f'{pref}{", ".join(lst)}{post}'

    @staticmethod
    def _list(lst: List, with_color=True):
        return PrettyLogger._iter(lst, with_color=with_color, pref='[', post=']')

    @staticmethod
    def _tuple(tpl: Tuple, with_color=True):
        return PrettyLogger._iter(tpl, with_color=with_color, pref='(', post=')')

    @staticmethod
    def _dict(d: Dict = None, with_color=True, pad_float: int = 5, sep=': ', for_path: bool = False, **kwargs) -> str:
        """
        Syntactic sugar for logging dict with coloring for console output
        """
        def _log_val(v):
            if isinstance(v, (dict, list, tuple)):
                return PrettyLogger.i(v, with_color=with_color)
            else:
                if is_float(v):  # Pad only normal, expected floats, intended for metric logging
                    if is_float(v, no_int=True, no_sci=True):
                        v = float(v)
                        if with_color:
                            return PrettyLogger.log(v, c='i', as_str=True, pad=pad_float)
                        else:
                            return f'{v:>{pad_float}}'
                    else:
                        return PrettyLogger.i(v) if with_color else v
                else:
                    return PrettyLogger.i(v) if with_color else v
        d = d or kwargs or dict()
        if for_path:
            assert not with_color  # sanity check
            sep = '='
        if with_color:
            sep = PrettyLogger.s(sep, c='m')
        pairs = (f'{k}{sep}{_log_val(v)}' for k, v in d.items())
        pref, post = '{', '}'
        if with_color:
            pref, post = PrettyLogger.s(pref, c='m'), PrettyLogger.s(post, c='m')
        return pref + ', '.join(pairs) + post


pl = PrettyLogger()


def hex2rgb(hx: str, normalize=False) -> Union[Tuple[int], Tuple[float]]:
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

        :param t: One of ['rgb`, `sty`]
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
    RESET = sty.rs.fg + sty.rs.bg + sty.rs.ef

    MyTheme.set_color_type('sty')
    yellow, green, blue, cyan, red, purple = (
        MyTheme.yellow, MyTheme.green, MyTheme.blue, MyTheme.cyan, MyTheme.red, MyTheme.purple
    )

    KW_TIME = '%(asctime)s'
    KW_MSG = '%(message)s'
    KW_LINENO = '%(lineno)d'
    KW_FNM = '%(filename)s'
    KW_FUNC_NM = '%(funcName)s'
    KW_NAME = '%(name)s'

    DEBUG = INFO = BASE = RESET
    WARN, ERR, CRIT = yellow, red, purple
    CRIT += sty.Style(sty.ef.bold)

    LVL_MAP = {  # level => (abbreviation, style)
        logging.DEBUG: ('DBG', DEBUG),
        logging.INFO: ('INFO', INFO),
        logging.WARNING: ('WARN', WARN),
        logging.ERROR: ('ERR', ERR),
        logging.CRITICAL: ('CRIT', CRIT)
    }

    def __init__(self, with_color=True, color_time=green):
        super().__init__()
        self.with_color = with_color

        sty_kw, reset = MyFormatter.blue, MyFormatter.RESET
        color_time = f'{color_time}{MyFormatter.KW_TIME}{sty_kw}|{reset}'

        def args2fmt(args_):
            if self.with_color:
                return color_time + self.fmt_meta(*args_) + f'{sty_kw}: {reset}{MyFormatter.KW_MSG}' + reset
            else:
                return f'{MyFormatter.KW_TIME}| {self.fmt_meta(*args_)}: {MyFormatter.KW_MSG}'

        self.formats = {level: args2fmt(args) for level, args in MyFormatter.LVL_MAP.items()}
        self.formatter = {
            lv: logging.Formatter(fmt, datefmt='%Y-%m-%d %H:%M:%S') for lv, fmt in self.formats.items()
        }

    def fmt_meta(self, meta_abv, meta_style=None):
        if self.with_color:
            return f'{MyFormatter.purple}[{MyFormatter.KW_NAME}]' \
               f'{MyFormatter.blue}::{MyFormatter.purple}{MyFormatter.KW_FUNC_NM}' \
               f'{MyFormatter.blue}::{MyFormatter.purple}{MyFormatter.KW_FNM}' \
               f'{MyFormatter.blue}:{MyFormatter.purple}{MyFormatter.KW_LINENO}' \
               f'{MyFormatter.blue}:{meta_style}{meta_abv}{MyFormatter.RESET}'
        else:
            return f'[{MyFormatter.KW_NAME}] {MyFormatter.KW_FUNC_NM}::{MyFormatter.KW_FNM}' \
                   f':{MyFormatter.KW_LINENO}, {meta_abv}'

    def format(self, entry):
        return self.formatter[entry.levelno].format(entry)


def get_logger(name: str, typ: str = 'stdout', file_path: str = None) -> logging.Logger:
    """
    :param name: Name of the logger
    :param typ: Logger type, one of [`stdout`, `file-write`]
    :param file_path: File path for file-write logging
    """
    assert typ in ['stdout', 'file-write']
    logger = logging.getLogger(f'{name} file write' if typ == 'file-write' else name)
    logger.handlers = []  # A crude way to remove prior handlers, ensure only 1 handler per logger
    logger.setLevel(logging.DEBUG)
    if typ == 'stdout':
        handler = logging.StreamHandler(stream=sys.stdout)  # stdout for my own coloring
    else:  # `file-write`
        if not file_path:
            raise ValueError(f'{pl.i(file_path)} must be specified for {pl.i("file-write")} logging')
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        handler = logging.FileHandler(file_path)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(MyFormatter(with_color=typ == 'stdout'))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


class MlPrettier:
    """
    My utilities for deep learning training logging
    """
    no_prefix = ('epoch', 'step')

    def __init__(
            self, ref: Dict[str, Any] = None, metric_keys: List[str] = None, no_prefix: Iterable[str] = no_prefix,
            with_color: bool = False
    ):
        """
        :param ref: Reference that are potentially needed
            i.e. for logging epoch/step, need the total #
        :param metric_keys: keys that are considered metric
            Will be logged in [0, 100]
        """
        self.ref = ref
        self.metric_keys = metric_keys or ['acc', 'recall', 'auc']
        self.no_prefix = no_prefix
        self.with_color = with_color

    def __call__(self, d: Union[str, Dict], val=None) -> Union[Any, Dict[str, Any]]:
        """
        :param d: If str, prettify a single value
            Otherwise, prettify a dict
        """
        is_dict = isinstance(d, dict)
        if not ((isinstance(d, str) and val is not None) or is_dict):
            raise ValueError('Either a key-value pair or a mapping is expected')
        if is_dict:
            d: Dict
            return {k: self.single(k, v) for k, v in d.items()}
        else:
            return self.single(d, val)

    def single(self, key: str = None, val: Any = None) -> Union[str, List[str], Dict[str, Any]]:
        """
        `val` processing is infered based on key
        """
        if key in ['step', 'epoch']:
            k = next(iter(k for k in self.ref.keys() if key in k))
            lim = self.ref[k]
            assert isinstance(val, (int, float))
            len_lim = len(str(lim))
            if isinstance(val, int):
                s_val = f'{val:>{len_lim}}'
            else:
                fmt = f'%{len_lim + 4}.3f'
                s_val = fmt % val
            if self.with_color:
                return f'{pl.i(s_val)}/{pl.i(lim)}'
            else:
                return f'{s_val}/{lim}'  # Pad integer
        elif 'loss' in key:
            return f'{round(val, 4):7.4f}'
        elif any(k in key for k in self.metric_keys):  # custom in-key-ratio metric
            def _single(v):
                return f'{round(v * 100, 2):6.2f}' if v is not None else '-'

            if isinstance(val, list):
                return [_single(v) for v in val]
            elif isinstance(val, dict):
                return {k: _single(v) for k, v in val.items()}
            else:
                return _single(val)
        elif 'learning_rate' in key or 'lr' in key:
            return f'{round(val, 7):.3e}'
        else:
            return val

    def should_add_split_prefix(self, key: str) -> bool:
        """
        Whether to add split prefix to the key
        """
        return key not in self.no_prefix

    def add_split_prefix(self, d: Dict[str, Any], split: str = None):
        if split is None:
            return d
        else:
            return {f'{split}/{k}' if self.should_add_split_prefix(k) else k: v for k, v in d.items()}


class MyProgressCallback(TrainerCallback):
    """
    My modification to the HF progress callback

    1. Effectively remove all logging, keep only the progress bar w.r.t. this callback
    2. Train tqdm for each epoch only
    3. Option to disable progress bar for evaluation

    Expects to start from whole epochs
    """
    def __init__(self, train_only: bool = False):
        """
        :param train_only: If true, disable progress bar for evaluation
        """
        self.training_bar = None
        self.prediction_bar = None

        self.train_only = train_only
        self.step_per_epoch = None
        self.current_step = None

    @staticmethod
    def _get_steps_per_epoch(state):
        assert state.max_steps % state.num_train_epochs == 0
        return state.max_steps // state.num_train_epochs

    @staticmethod
    def _get_curr_epoch(state, is_eval: bool = False) -> str:
        n_ep = int(state.epoch)
        if not is_eval:  # heuristic judging by the eval #epoch shown
            n_ep += 1

        return MlPrettier(ref=dict(epoch=state.num_train_epochs), with_color=True)('epoch', n_ep)

    def on_epoch_begin(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            if not self.step_per_epoch:
                self.step_per_epoch = MyProgressCallback._get_steps_per_epoch(state)
            ep = MyProgressCallback._get_curr_epoch(state)
            self.training_bar = tqdm(total=self.step_per_epoch, desc=f'Train Epoch {ep}', unit='ba')
        self.current_step = 0

    def on_train_begin(self, args, state, control, **kwargs):
        pass

    def on_epoch_end(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            self.training_bar.close()
            self.training_bar = None

    def on_step_end(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            self.training_bar.update(1)

    def on_prediction_step(self, args, state, control, eval_dataloader=None, **kwargs):
        if not self.train_only:
            if state.is_local_process_zero and isinstance(eval_dataloader.dataset, Sized):
                if self.prediction_bar is None:
                    ep = MyProgressCallback._get_curr_epoch(state, is_eval=True)
                    desc = f'Eval Epoch {ep}'
                    self.prediction_bar = tqdm(
                        desc=desc, total=len(eval_dataloader), leave=self.training_bar is None, unit='ba'
                    )
                self.prediction_bar.update(1)

    def on_evaluate(self, args, state, control, **kwargs):
        if not self.train_only:
            if state.is_local_process_zero:
                if self.prediction_bar is not None:
                    self.prediction_bar.close()
                self.prediction_bar = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero and self.training_bar is not None:
            _ = logs.pop("total_flos", None)

    def on_train_end(self, args, state, control, **kwargs):
        pass

    @staticmethod
    def get_current_progress_bar(trainer: Trainer):
        """
        Intended for adding per-step metrics to the progress bar during HF training

        This is a hack,
            since HF API don't support per-step callback not to mention exposing those metrics to the progress bar
        """
        callback = next(cb for cb in trainer.callback_handler.callbacks if isinstance(cb, MyProgressCallback))
        return callback.training_bar if trainer.model.training else callback.prediction_bar


class LogStep:
    """
    My typical terminal, file-write, tqdm logging for a single step
    """
    def __init__(
            self, trainer: Trainer = None, prettier: MlPrettier = None,
            logger: logging.Logger = None, file_logger: logging.Logger = None,
            tb_writer: SummaryWriter = None
    ):
        if not trainer:
            raise ValueError(f'A {pl.i("trainer")} is required')
        if not hasattr(trainer, 'with_tqdm'):
            raise ValueError(f'Trainer ill-formed: A custom {pl.i("with_tqdm")} field is needed')
        self.trainer = trainer
        self.prettier = prettier
        self.logger = logger
        self.file_logger = file_logger
        self.tb_writer = tb_writer

    def _should_add(self, key: str) -> bool:
        return self.prettier.should_add_split_prefix(key) if self.prettier else True

    def __call__(self, d_log: Dict, training: bool = None, to_console: bool = True):
        training = training or self.trainer.model.training
        d_log_write = self.prettier(d_log) if self.prettier else d_log

        if self.tb_writer:
            tb_step = d_log.get('step') if training else d_log.get('epoch')
            pref = 'train' if training else 'eval'
            for k, v in d_log.items():
                if self._should_add(k):
                    self.tb_writer.add_scalar(tag=f'{pref}/{k}', scalar_value=v, global_step=tb_step)

        if self.trainer.with_tqdm:  # a custom field I added
            pbar = MyProgressCallback.get_current_progress_bar(self.trainer)
            if pbar:
                tqdm_kws = {k: pl.i(v) for k, v in d_log_write.items() if self._should_add(k)}
                pbar.set_postfix(tqdm_kws)
        if to_console:
            if self.logger:
                self.logger.info(pl.i(d_log))

        if self.file_logger:
            self.file_logger.info(pl.nc(d_log))


class CheckArg:
    """
    An easy, readable interface for checking string arguments as effectively enums

    Intended for high-level arguments instead of actual data processing as not as efficient

    Raise errors when common arguments don't match the expected values
    """
    logger = get_logger('Arg Checker')

    def check_mismatch(self, display_name: str, val: str, accepted_values: List[str]):
        if self.verbose:
            d_log = dict(val=val, accepted_values=accepted_values)
            CheckArg.logger.info(f'Checking {pl.i(display_name)} w/ {pl.i(d_log)}... ')
        if val not in accepted_values:
            raise ValueError(f'Unexpected {pl.i(display_name)}: '
                             f'expect one of {pl.i(accepted_values)}, got {pl.i(val)}')

    def __init__(self, ignore_none: bool = True, verbose: bool = False):
        """
        :param ignore_none: If true, arguments passed in as `None` will not raise error
        :param verbose: If true, logging are print to console
        """
        self.d_name2func = dict()
        self.ignore_none = ignore_none
        self.verbose = verbose

    def __call__(self, **kwargs):
        for k, v in kwargs.items():
            if self.ignore_none and v is None:
                if self.verbose:
                    CheckArg.logger.warning(f'Argument {pl.i(k)} is {pl.i("None")} and ignored')
                continue
            self.d_name2func[k](v)

    def cache_mismatch(self, display_name: str, attr_name: str, accepted_values: List[str]):
        self.d_name2func[attr_name] = lambda x: self.check_mismatch(display_name, x, accepted_values)


ca = CheckArg()
ca.cache_mismatch(  # See `stefutil::plot.py`
    'Bar Plot Orientation', attr_name='bar_orient', accepted_values=['v', 'h', 'vertical', 'horizontal']
)


def now(as_str=True, for_path=False, fmt: str = 'full') -> Union[datetime.datetime, str]:
    """
    # Considering file output path
    :param as_str: If true, returns string; otherwise, returns datetime object
    :param for_path: If true, the string returned is formatted as intended for file system path
        relevant only when as_str is True
    :param fmt: One of [`full`, `date`, `short-date`]
        relevant only when as_str is True
    """
    d = datetime.datetime.now()
    if as_str:
        ca.check_mismatch('Date Format', fmt, ['full', 'date', 'short-date'])
        if fmt == 'full':
            fmt_tm = '%Y-%m-%d_%H-%M-%S' if for_path else '%Y-%m-%d %H:%M:%S.%f'
        else:
            fmt_tm = '%Y-%m-%d'
        ret = d.strftime(fmt_tm)
        if fmt == 'short-date':  # year in 2-digits
            ret = ret[2:]
        return ret
    else:
        return d


if __name__ == '__main__':
    # lg = get_logger('test')
    # lg.info('test')

    def check_log_lst():
        lst = ['sda', 'asd']
        print(pl.i(lst))
        # with open('test-logi.txt', 'w') as f:
        #     f.write(pl.nc(lst))
    # check_log_lst()

    def check_log_tup():
        tup = ('sda', 'asd')
        print(pl.i(tup))
    # check_log_tup()

    def check_logi():
        d = dict(a=1, b=2)
        print(pl.i(d))
    # check_logi()

    def check_nested_log_dict():
        d = dict(a=1, b=2, c=dict(d=3, e=4, f=['as', 'as']))
        mic(d)
        print(pl.i(d))
        print(pl.nc(d))
        mic(pl.i(d), pl.nc(d))
    # check_nested_log_dict()

    def check_logger():
        logger = get_logger('blah')
        logger.info('should appear once')
    # check_logger()

    def check_now():
        mic(now(fmt='full'))
        mic(now(fmt='date'))
        mic(now(fmt='short-date'))
    # check_now()

    def check_ca():
        ori = 'v'
        ca(bar_orient=ori)
    # check_ca()

    def check_ca_warn():
        ca_ = CheckArg(verbose=True)
        ca_.cache_mismatch(display_name='Disp Test', attr_name='test', accepted_values=['a', 'b'])
        ca_(test='a')
        ca_(test=None)
    # check_ca_warn()

    def check_time_delta():
        import datetime
        now_ = datetime.datetime.now()
        last_day = now_ - datetime.timedelta(days=1, hours=1, minutes=1, seconds=1)
        mic(now_, last_day)
        diff = now_ - last_day
        mic(diff, fmt_delta(diff))
    check_time_delta()
