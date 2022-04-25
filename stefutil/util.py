import os
import re
import sys
import json
import math
import time
import pathlib
import logging
import datetime
import itertools
import configparser
import concurrent.futures
from os.path import join as os_join
from typing import (
    Tuple, List, Dict,
    TypeVar, Iterable, Callable,
    Any, Union
)
from pygments import highlight, lexers, formatters
from functools import reduce
from collections import OrderedDict

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import colorama
import sty

from stefutil.check_args import ca


__all__ = [
    'nan', 'LN_KWARGS',  # constants
    'get', 'set_',  # recursive dictionary syntactic sugar
    'StefConfig', 'StefUtil',  # Project & project file structure related
    'get_hostname', 'stem',  # os-related
    'is_float', 'get_substr_indices',  # primitive & numpy manipulation
    'chain_its', 'join_it', 'group_n', 'lst2uniq_ids', 'np_index',  # data structure manipulation
    'conc_map', 'batched_conc_map',  # concurrency
    # prettier & prettier logging
    'fmt_num', 'fmt_sizeof', 'fmt_dt', 'nth_sig_digit', 'now',
    'log', 'log_s', 'logi', 'log_dict', 'log_dict_nc', 'log_dict_id', 'log_dict_pg', 'log_dict_p',
    'hex2rgb', 'MyTheme', 'MyFormatter', 'get_logger',
    'profile_runtime',  # timing
    # plot; see also `StefUtil.save_fig`
    'save_fig', 'change_bar_width', 'vals2colors', 'set_color_bar', 'barplot',
    'model_param_size', 'get_model_num_trainable_parameter', 'is_on_colab'  # machine learning
]


pd.set_option('expand_frame_repr', False)
pd.set_option('display.precision', 2)
pd.set_option('max_colwidth', 40)
pd.set_option('display.max_columns', None)

plt.rc('figure', figsize=(16, 9))
plt.rc('figure.constrained_layout', use=True)
plt.rc('text.latex', preamble='\n'.join([
    r'\usepackage{nicefrac}',
    r'\usepackage{helvet}',
    r'\usepackage{sansmath}',  # render math sans-serif
    r'\sansmath'
]))
sns.set_style('darkgrid')
sns.set_context(rc={'grid.linewidth': 0.5})


nan = float('nan')
T = TypeVar('T')
K = TypeVar('K')
LN_KWARGS = dict(marker='o', ms=0.3, lw=0.25)  # matplotlib line plot default args


def get(dic: Dict, ks: str):
    """
    :param dic: Potentially multi-level dictionary
    :param ks: Potentially `.`-separated keys
    """
    ks = ks.split('.')
    return reduce(lambda acc, elm: acc[elm], ks, dic)


def set_(dic, ks, val):
    ks = ks.split('.')
    node = reduce(lambda acc, elm: acc[elm], ks[:-1], dic)
    node[ks[-1]] = val


class StefConfig:
    """
    the one-stop place for package-level constants, expects a json file
    """
    def __init__(self, config_file: str):
        self.config_file = config_file
        with open(config_file, 'r') as f:
            self.d = json.load(f)

    def __call__(self, keys: str):
        """
        Retrieves the queried attribute value from the config file

        Loads the config file on first call.
        """
        return get(self.d, keys)


def it_keys(dic, prefix=''):
    """
    :return: Generator for all potentially-nested keys
    """
    def _full(k_):
        return k_ if prefix == '' else f'{prefix}.{k_}'
    for k, v in dic.items():
        if isinstance(v, dict):
            for k__ in it_keys(v, prefix=_full(k)):
                yield k__
        else:
            yield _full(k)


class StefUtil:
    """
    Effectively curried functions with my enforced project & dataset structure
        Pass in file paths
    """
    def __init__(self, base_path: str, proj_dir: str, pkg_nm: str, dset_dir: str):
        """
        :param base_path: Root directory that contains a directory for project & a directory for data
        :param proj_dir: Project directory name
        :param pkg_nm: Project main source files package directory name
        :param dset_dir: Data directory name
        """
        self.base_path = base_path
        self.proj_dir = proj_dir
        self.pkg_nm = pkg_nm
        self.dset_dir = dset_dir

        self.plot_dir = os_join(self.base_path, self.proj_dir, 'plots')

    def save_fig(self, title, save=True):
        if save:
            fnm = f'{title}, {now(for_path=True)}.png'
            plt.savefig(os_join(self.plot_dir, fnm), dpi=300)


def get_hostname() -> str:
    return os.uname().nodename


def stem(path_: Union[str, pathlib.Path], ext=False) -> str:
    """
    :param path_: A potentially absolute path to a file
    :param ext: If True, file extensions is preserved
    :return: The file name, without parent directories
    """
    return os.path.basename(path_) if ext else pathlib.Path(path_).stem


def is_float(x: Any, no_int=False, no_sci=False) -> bool:
    try:
        is_sci = isinstance(x, str) and 'e' in x.lower()
        f = float(x)
        is_int = f.is_integer()
        out = True
        if no_int:
            out = out & (not is_int)
        if no_sci:
            out = out & (not is_sci)
        return out
    except ValueError:
        return False

def clean_whitespace(s: str):
    if not hasattr(clean_whitespace, 'pattern_space'):
        clean_whitespace.pattern_space = re.compile(r'\s+')
    return clean_whitespace.pattern_space.sub(' ', s).strip()


def get_substr_indices(s: str, s_sub: str) -> List[int]:
    s_sub = re.escape(s_sub)
    return [m.start() for m in re.finditer(s_sub, s)]


def chain_its(its: Iterable[Iterable[T]]) -> Iterable[T]:
    """
    Chain multiple iterables
    """
    out = itertools.chain()
    for it in its:
        out = itertools.chain(out, it)
    return out


def join_it(it: Iterable[T], sep: T) -> Iterable[T]:
    """
    Generic join elements with separator element, like `str.join`
    """
    it = iter(it)

    curr = next(it, None)
    if curr is not None:
        yield curr
        curr = next(it, None)
    while curr is not None:
        yield sep
        yield curr
        curr = next(it, None)


def group_n(it: Iterable[T], n: int) -> Iterable[Tuple[T]]:
    """
    Slice iterable into groups of size n (last group included) by iteration order
    """
    # Credit: https://stackoverflow.com/a/8991553/10732321
    it = iter(it)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def conc_map(fn: Callable[[T], K], it: Iterable[T]) -> Iterable[K]:
    """
    Wrapper for `concurrent.futures.map`

    :param fn: A function
    :param it: A list of elements
    :return: Iterator of `lst` elements mapped by `fn` with concurrency
    """
    with concurrent.futures.ThreadPoolExecutor() as executor:
        return executor.map(fn, it)


def batched_conc_map(
        fn: Callable[[Tuple[List[T], int, int]], K], lst: List[T], n_worker: int = os.cpu_count()
) -> List[K]:
    """
    Batched concurrent mapping, map elements in list in batches

    :param fn: A map function that operates on a batch/subset of `lst` elements,
        given inclusive begin & exclusive end indices
    :param lst: A list of elements to map
    :param n_worker: Number of concurrent workers
    """
    n: int = len(lst)
    if n_worker > 1 and n > n_worker * 4:  # factor of 4 is arbitrary, otherwise not worse the overhead
        preprocess_batch = round(n / n_worker / 2)
        strts: List[int] = list(range(0, n, preprocess_batch))
        ends: List[int] = strts[1:] + [n]  # inclusive begin, exclusive end
        lst_out = []
        for lst_ in conc_map(lambda args_: fn(*args_), [(lst, s, e) for s, e in zip(strts, ends)]):  # Expand the args
            lst_out.extend(lst_)
        return lst_out
    else:
        args = lst, 0, n
        return fn(*args)


def lst2uniq_ids(lst: List[T]) -> List[int]:
    """
    Each unique element in list assigned an unique id, in increasing order of iteration
    """
    elm2id = {v: k for k, v in enumerate(OrderedDict.fromkeys(lst))}
    return [elm2id[e] for e in lst]


def np_index(arr, idx):
    return np.where(arr == idx)[0][0]


def fmt_num(num: Union[float, int], suffix: str = '') -> str:
    """
    Convert number to human-readable format, in e.g. Thousands, Millions
    """
    # if not hasattr(fmt_num, 'posts'):
    #     fmt_num.posts = ['', 'K', 'M', 'B', 'T']
    # n = float(n)
    # idx_ = max(0, min(len(fmt_num.posts) - 1, int(math.floor(0 if n == 0 else math.log10(abs(n)) / 3))))
    # return '{:.0f}{}'.format(n / 10 ** (3 * idx_), fmt_num.posts[idx_])
    for unit in ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Y', suffix)


def fmt_sizeof(num: int, suffix='B') -> str:
    """ Converts byte size to human-readable format """
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


def fmt_dt(secs: Union[int, float, datetime.timedelta]) -> str:
    if isinstance(secs, datetime.timedelta):
        secs = secs.seconds + (secs.microseconds/1e6)
    if secs >= 86400:
        d = secs // 86400  # // floor division
        return f'{round(d)}d{fmt_dt(secs-d*86400)}'
    elif secs >= 3600:
        h = secs // 3600
        return f'{round(h)}h{fmt_dt(secs-h*3600)}'
    elif secs >= 60:
        m = secs // 60
        return f'{round(m)}m{fmt_dt(secs-m*60)}'
    else:
        return f'{round(secs)}s'


def nth_sig_digit(flt: float, n: int = 1) -> float:
    """
    :return: first n-th significant digit of `sig_d`
    """
    return float('{:.{p}g}'.format(flt, p=n))



def now(as_str=True, for_path=False) -> Union[datetime.datetime, str]:
    """
    # Considering file output path
    :param as_str: If true, returns string; otherwise, returns datetime object
    :param for_path: If true, the string returned is formatted as intended for file system path
    """
    d = datetime.datetime.now()
    fmt = '%Y-%m-%d_%H-%M-%S' if for_path else '%Y-%m-%d %H:%M:%S'
    return d.strftime(fmt) if as_str else d


def log(s, c: str = 'log', c_time='green', as_str=False, bold=False):
    """
    Prints `s` to console with color `c`
    """
    if not hasattr(log, 'reset'):
        log.reset = colorama.Fore.RESET + colorama.Back.RESET + colorama.Style.RESET_ALL
    if not hasattr(log, 'd'):
        log.d = dict(
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
    if c in log.d:
        c = log.d[c]
    if bold:
        c += colorama.Style.BRIGHT
    if as_str:
        return f'{c}{s}{log.reset}'
    else:
        print(f'{c}{log(now(), c=c_time, as_str=True)}| {s}{log.reset}')


def log_s(s, c, bold: bool = False):
    return log(s, c=c, as_str=True, bold=bold)


def logi(s):
    """
    Syntactic sugar for logging `info` as string
    """
    return log_s(s, c='i')


def log_dict(d: Dict = None, with_color=True, pad_float: int = 5, sep=': ', **kwargs) -> str:
    """
    Syntactic sugar for logging dict with coloring for console output
    """
    def _log_val(v):
        if isinstance(v, dict):
            return log_dict(v, with_color=with_color)
        else:
            # Pad only normal, expected floats, intended for metric logging
            if not isinstance(v, (tuple, list)) and is_float(v):
                if is_float(v, no_int=True, no_sci=True):
                    v = float(v)
                    return log(v, c='i', as_str=True, pad=pad_float) if with_color else f'{v:>{pad_float}}'
                else:
                    return logi(v) if with_color else v
            else:
                return logi(v) if with_color else v
    d = d or kwargs or dict()
    pairs = (f'{k}{sep}{_log_val(v)}' for k, v in d.items())
    pref = log_s('{', c='m') if with_color else '{'
    post = log_s('}', c='m') if with_color else '}'
    return pref + ', '.join(pairs) + post


def log_dict_nc(d: Dict = None, **kwargs) -> str:
    """
    Syntactic sugar for no color
    """
    return log_dict(d, with_color=False, **kwargs)


def log_dict_id(d: Dict) -> str:
    """
    Indented dict
    """
    return json.dumps(d, indent=4)


def log_dict_pg(d: Dict) -> str:
    """
    prettier dict colored by `pygments` and with indent
    """
    return highlight(log_dict_id(d), lexers.JsonLexer(), formatters.TerminalFormatter())


def log_dict_p(d: Dict, **kwargs) -> str:
    """
    a compact, one-line str for dict

    Intended as part of filename
    """
    return log_dict(d, with_color=False, sep='=', **kwargs)


def hex2rgb(hx: str) -> Union[Tuple[int], Tuple[float]]:
    # Modified from https://stackoverflow.com/a/62083599/10732321
    if not hasattr(hex2rgb, 'regex'):
        hex2rgb.regex = re.compile(r'#[a-fA-F0-9]{3}(?:[a-fA-F0-9]{3})?$')
    m = hex2rgb.regex.match(hx)
    assert m is not None
    if len(hx) <= 4:
        return tuple(int(hx[i]*2, 16) for i in range(1, 4))
    else:
        return tuple(int(hx[i:i+2], 16) for i in range(1, 7, 2))


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
    KW_FUNCNM = '%(funcName)s'
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
        color_time = f'{color_time}{MyFormatter.KW_TIME}{sty_kw}| {reset}'

        def args2fmt(args_):
            if self.with_color:
                return color_time + self.fmt_meta(*args_) + f'{sty_kw} - {reset}{MyFormatter.KW_MSG}' + reset
            else:
                return f'{MyFormatter.KW_TIME}| {self.fmt_meta(*args_)} - {MyFormatter.KW_MSG}'

        self.formats = {level: args2fmt(args) for level, args in MyFormatter.LVL_MAP.items()}
        self.formatter = {
            lv: logging.Formatter(fmt, datefmt='%Y-%m-%d %H:%M:%S') for lv, fmt in self.formats.items()
        }

    def fmt_meta(self, meta_abv, meta_style=None):
        if self.with_color:
            return f'{MyFormatter.purple}[{MyFormatter.KW_NAME}]' \
               f'{MyFormatter.blue}::{MyFormatter.purple}{MyFormatter.KW_FUNCNM}' \
               f'{MyFormatter.blue}::{MyFormatter.purple}{MyFormatter.KW_FNM}' \
               f'{MyFormatter.blue}:{MyFormatter.purple}{MyFormatter.KW_LINENO}' \
               f'{MyFormatter.blue}, {meta_style}{meta_abv}{MyFormatter.RESET}'
        else:
            return f'[{MyFormatter.KW_NAME}] {MyFormatter.KW_FUNCNM}::{MyFormatter.KW_FNM}' \
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
    else:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        handler = logging.FileHandler(file_path)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(MyFormatter(with_color=typ == 'stdout'))
    logger.addHandler(handler)
    return logger


def profile_runtime(callback: Callable, sleep: Union[float, int] = None):
    import cProfile
    import pstats
    profiler = cProfile.Profile()
    profiler.enable()
    callback()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    if sleep != 0:    # Sometimes, the top rows in `print_states` are now shown properly
        time.sleep(sleep)
    stats.print_stats()


def change_bar_width(ax, width: float = 0.5, orient: str = 'v'):
    """
    Modifies the bar width of a matplotlib bar plot

    Credit: https://stackoverflow.com/a/44542112/10732321
    """
    ca(orient=orient)
    is_vert = orient in ['v', 'vertical']
    for patch in ax.patches:
        current_width = patch.get_width() if is_vert else patch.get_height()
        diff = current_width - width
        patch.set_width(width) if is_vert else patch.set_height(width)
        patch.set_x(patch.get_x() + diff * .5) if is_vert else patch.set_y(patch.get_y() + diff * .5)


def vals2colors(vals: Iterable[float], color_palette: str = 'Spectral_r') -> Iterable:
    """
    Map an iterable of values to corresponding colors given a color map
    """
    vals = np.asarray(vals)
    cmap = sns.color_palette(color_palette, as_cmap=True)
    mi, ma = np.min(vals), np.max(vals)
    norm = (vals - mi) / (ma - mi)
    return cmap(norm)


def set_color_bar(vals, ax, color_palette: str = 'Spectral_r', orientation: str = 'vertical'):
    """
    Set give axis to show the color bard
    """
    vals = np.asarray(vals)
    norm = plt.Normalize(vmin=np.min(vals), vmax=np.max(vals))
    sm = plt.cm.ScalarMappable(cmap=color_palette, norm=norm)
    sm.set_array([])
    plt.sca(ax)
    plt.grid(False)
    plt.colorbar(sm, cax=ax, orientation=orientation)
    # plt.xlabel('colorbar')  # doesn't seem to work


def barplot(
        x: Iterable[str], y: Iterable[float], orient: str = 'v', with_value: bool = True, width: float = 0.5,
        xlabel: str = None, ylabel: str = None,
        ax=None, palette=None, **kwargs
):
    ca(orient=orient)
    df = pd.DataFrame([dict(x=x_, y=y_) for x_, y_ in zip(x, y)])
    cat = CategoricalDtype(categories=x, ordered=True)  # Enforce ordering in plot
    df['x'] = df['x'].astype(cat, copy=False)
    is_vert = orient in ['v', 'vertical']
    x, y = ('x', 'y') if is_vert else ('y', 'x')
    if ax:
        kwargs['ax'] = ax
    if palette is not None:
        kwargs['palette'] = palette
    ax = sns.barplot(data=df, x=x, y=y, **kwargs)
    if with_value:
        ax.bar_label(ax.containers[0])
    if width:
        change_bar_width(ax, width, orient=orient)
    ax.set_xlabel(xlabel) if is_vert else ax.set_ylabel(xlabel)  # if None just clears the label
    ax.set_ylabel(ylabel) if is_vert else ax.set_xlabel(ylabel)



def get_torch_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def model_param_size(m: torch.nn.Module, as_str=True) -> Union[int, str]:
    num = sum(p.numel() for p in m.parameters())
    return fmt_num(num) if as_str else num

def get_model_num_trainable_parameter(model: torch.nn.Module, readable: bool = True) -> Union[int, str]:
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return readable_int(n) if readable else n


def is_on_colab() -> bool:
    return 'google.colab' in sys.modules


__all__ = [
    'LN_KWARGS', 'nan',
    'vars_', 'get', 'set_', 'it_keys', 'config',
    'compress', 'flatten', 'list_split', 'join_its', 'group_n', 'sample', 'conc_map', 'batched_conc_map',
    'readable_int', 'now', 'fmt_time', 'sec2mmss', 'round_up_1digit', 'profile_runtime',
    'clip', 'np_index', 'clean_whitespace', 'stem', 'list_is_same_elms', 'save_fig', 'get_plot_path', 'read_pickle',
    'is_on_colab', 'get_model_num_trainable_parameter',
    'log', 'log_s', 'logi', 'is_float', 'log_dict', 'log_dict_nc', 'log_dict_id', 'log_dict_pg', 'log_dict_p',
    'hex2rgb', 'MyTheme', 'MyFormatter', 'get_logger',
    'RecurseLimit'
]




def list_is_same_elms(lst: List[T]) -> bool:
    return all(l == lst[0] for l in lst)


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


def get(dic, ks):
    """
    :param dic: Potentially multi-level dictionary
    :param ks: Potentially `.`-separated keys
    """
    ks = ks.split('.')
    return reduce(lambda acc, elm: acc[elm], ks, dic)


def set_(dic, ks, val):
    ks = ks.split('.')
    node = reduce(lambda acc, elm: acc[elm], ks[:-1], dic)
    node[ks[-1]] = val


def it_keys(dic, prefix=''):
    """
    :return: Generator for all potentially-nested keys
    """
    def _full(k_):
        return k_ if prefix == '' else f'{prefix}.{k_}'
    for k, v in dic.items():
        if isinstance(v, dict):
            for k__ in it_keys(v, prefix=_full(k)):
                yield k__
        else:
            yield _full(k)


def config(attr):
    """
    Retrieves the queried attribute value from the config file.

    Loads the config file on first call.
    """
    if not hasattr(config, 'config'):
        with open(os.path.join(PATH_BASE, DIR_PROJ, PKG_NM, 'util', 'config.json'), 'r') as f:
            config.config = json.load(f)
    return get(config.config, attr)


def readable_int(num: int, suffix: str = '') -> str:
    """
    Converts (potentially large) integer to human-readable format
    """
    for unit in ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Y', suffix)


def now(as_str=True, for_path=False):
    """
    # Considering file output path
    :param as_str: If true, returns string; otherwise, returns datetime object
    :param for_path: If true, the string returned is formatted as intended for file system path
    """
    d = datetime.datetime.now()
    fmt = '%Y-%m-%d_%H-%M-%S' if for_path else '%Y-%m-%d %H:%M:%S'
    return d.strftime(fmt) if as_str else d


def get_plot_path() -> str:
    return os.path.join(PATH_BASE, DIR_PROJ, 'plot')


def save_fig(title, save=True):
    plt_path = get_plot_path()
    os.makedirs(plt_path, exist_ok=True)
    if save:
        fnm = f'{title}, {now(for_path=True)}.png'
        plt.savefig(os.path.join(plt_path, fnm), dpi=300)


def read_pickle(fnm):
    objects = []
    with (open(fnm, 'rb')) as f:
        while True:
            try:
                objects.append(pickle.load(f))
            except EOFError:
                break
    return objects


def is_on_colab() -> bool:
    return 'google.colab' in sys.modules


def get_model_num_trainable_parameter(model: torch.nn.Module, readable: bool = True) -> Union[int, str]:
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return readable_int(n) if readable else n


def fmt_time(secs: Union[int, float, datetime.timedelta]):
    if isinstance(secs, datetime.timedelta):
        secs = secs.seconds + (secs.microseconds/1e6)
    if secs >= 86400:
        d = secs // 86400  # // floor division
        return f'{round(d)}d{fmt_time(secs - d * 86400)}'
    elif secs >= 3600:
        h = secs // 3600
        return f'{round(h)}h{fmt_time(secs - h * 3600)}'
    elif secs >= 60:
        m = secs // 60
        return f'{round(m)}m{fmt_time(secs - m * 60)}'
    else:
        return f'{round(secs)}s'


def sec2mmss(sec: int) -> str:
    return str(datetime.timedelta(seconds=sec))[2:]


def round_up_1digit(num: int):
    d = math.floor(math.log10(num))
    fact = 10**d
    return math.ceil(num/fact) * fact


def profile_runtime(callback: Callable, sleep: Union[float, int] = None):
    import cProfile
    import pstats
    profiler = cProfile.Profile()
    profiler.enable()
    callback()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    if sleep:    # Sometimes, the top rows in `print_states` are now shown properly
        time.sleep(sleep)
    stats.print_stats()


def compress(lst: List[T]) -> List[Tuple[T, int]]:
    """
    :return: A compressed version of `lst`, as 2-tuple containing the occurrence counts
    """
    if not lst:
        return []
    return ([(lst[0], len(list(itertools.takewhile(lambda elm: elm == lst[0], lst))))]
            + compress(list(itertools.dropwhile(lambda elm: elm == lst[0], lst))))


def flatten(lsts):
    """ Flatten list of [list of elements] to list of elements """
    return sum(lsts, [])


def list_split(lst: List[T], call: Callable[[T], bool]) -> List[List[T]]:
    """
    :return: Split a list by locations of elements satisfying a condition
    """
    return [list(g) for k, g in itertools.groupby(lst, call) if k]


def join_its(its: Iterable[Iterable[T]]) -> Iterable[T]:
    out = itertools.chain()
    for it in its:
        out = itertools.chain(out, it)
    return out


def group_n(it: Iterable[T], n: int) -> Iterable[Tuple[T]]:
    # Credit: https://stackoverflow.com/a/8991553/10732321
    it = iter(it)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def sample(d: Dict[K, Union[float, Any]]) -> K:
    """
    Sample a key from a dict based on confidence score as value
        Keys with confidence evaluated to false are ignored

    Internally uses `torch.multinomial`
    """
    d_keys = {k: v for k, v in d.items() if v}  # filter out `None`s
    keys, weights = zip(*d_keys.items())
    return keys[torch.multinomial(torch.tensor(weights), 1, replacement=True).item()]


def conc_map(fn: Callable[[T], K], it: Iterable[T], with_tqdm=False) -> Iterable[K]:
    """
    Wrapper for `concurrent.futures.map`

    :param fn: A function
    :param it: A list of elements
    :return: Iterator of `lst` elements mapped by `fn` with concurrency
    :param with_tqdm: If true, progress bar is shown
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        ret = list(tqdm(executor.map(fn, it), total=len(list(it)))) if with_tqdm else executor.map(fn, it)
    return ret


def batched_conc_map(
        fn: Callable[[Tuple[List[T], int, int]], K], lst: List[T], n_worker: int = os.cpu_count(),
        batch_size: int = None,
        with_tqdm: bool = False  # TODO: doesn't seem to work as expected
) -> List[K]:
    """
    Batched concurrent mapping, map elements in list in batches

    :param fn: A map function that operates on a batch/subset of `lst` elements,
        given inclusive begin & exclusive end indices
    :param lst: A list of elements to map
    :param n_worker: Number of concurrent workers
    :param batch_size: Number of elements for each sub-process worker
        Inferred based on number of workers if not given
    :param with_tqdm: If true, progress bar is shown
    """
    n: int = len(lst)
    if (n_worker > 1 and n > n_worker * 4) or batch_size:  # factor of 4 is arbitrary, otherwise not worse the overhead
        preprocess_batch = batch_size or round(n / n_worker / 2)
        strts: List[int] = list(range(0, n, preprocess_batch))
        ends: List[int] = strts[1:] + [n]  # inclusive begin, exclusive end
        lst_out = []
        # Expand the args
        map_out = conc_map(lambda args_: fn(*args_), [(lst, s, e) for s, e in zip(strts, ends)], with_tqdm=with_tqdm)
        for lst_ in map_out:
            lst_out.extend(lst_)
        return lst_out
    else:
        args = lst, 0, n
        return fn(*args)


def log(s, c: str = 'log', c_time='green', as_str=False, pad: int = None):
    """
    Prints `s` to console with color `c`
    """
    if not hasattr(log, 'reset'):
        log.reset = colorama.Fore.RESET + colorama.Back.RESET + colorama.Style.RESET_ALL
    if not hasattr(log, 'd'):
        log.d = dict(
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
    if c in log.d:
        c = log.d[c]
    if as_str:
        return f'{c}{s:>{pad}}{log.reset}' if pad is not None else f'{c}{s}{log.reset}'
    else:
        print(f'{c}{log(now(), c=c_time, as_str=True)}| {s}{log.reset}')


def log_s(s, c):
    return log(s, c=c, as_str=True)


def logi(s):
    """
    Syntactic sugar for logging `info` as string
    """
    return log_s(s, c='i')


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


def log_dict(d: Dict = None, with_color=True, pad_float: int = 5, sep=': ', **kwargs) -> str:
    """
    Syntactic sugar for logging dict with coloring for console output
    """
    def _log_val(v):
        if isinstance(v, dict):
            return log_dict(v, with_color=with_color)
        else:
            if is_float(v):  # Pad only normal, expected floats, intended for metric logging
                if is_float(v, no_int=True, no_sci=True):
                    v = float(v)
                    return log(v, c='i', as_str=True, pad=pad_float) if with_color else f'{v:>{pad_float}}'
                else:
                    return logi(v) if with_color else v
            else:
                return logi(v) if with_color else v
    d = d or kwargs or dict()
    pairs = (f'{k}{sep}{_log_val(v)}' for k, v in d.items())
    pref = log_s('{', c='m') if with_color else '{'
    post = log_s('}', c='m') if with_color else '}'
    return pref + ', '.join(pairs) + post


def log_dict_nc(d: Dict, **kwargs) -> str:
    return log_dict(d, with_color=False, **kwargs)


def log_dict_id(d: Dict) -> str:
    """
    Indented dict
    """
    return json.dumps(d, indent=4)


def log_dict_pg(d: Dict) -> str:
    return highlight(log_dict_id(d), lexers.JsonLexer(), formatters.TerminalFormatter())


def log_dict_p(d: Dict, **kwargs) -> str:
    """
    for path
    """
    return log_dict(d, with_color=False, sep='=', **kwargs)


def hex2rgb(hx: str, normalize=False) -> Union[Tuple[int], Tuple[float]]:
    # Modified from https://stackoverflow.com/a/62083599/10732321
    if not hasattr(hex2rgb, 'regex'):
        hex2rgb.regex = re.compile(r'#[a-fA-F0-9]{3}(?:[a-fA-F0-9]{3})?$')
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
    KW_FUNCNM = '%(funcName)s'
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
                return color_time + self.fmt_meta(*args_) + f'{sty_kw} - {reset}{MyFormatter.KW_MSG}' + reset
            else:
                return f'{MyFormatter.KW_TIME}| {self.fmt_meta(*args_)} - {MyFormatter.KW_MSG}'

        self.formats = {level: args2fmt(args) for level, args in MyFormatter.LVL_MAP.items()}
        self.formatter = {
            lv: logging.Formatter(fmt, datefmt='%Y-%m-%d %H:%M:%S') for lv, fmt in self.formats.items()
        }

    def fmt_meta(self, meta_abv, meta_style=None):
        if self.with_color:
            return f'{MyFormatter.purple}[{MyFormatter.KW_NAME}]' \
               f'{MyFormatter.blue}::{MyFormatter.purple}{MyFormatter.KW_FUNCNM}' \
               f'{MyFormatter.blue}::{MyFormatter.purple}{MyFormatter.KW_FNM}' \
               f'{MyFormatter.blue}:{MyFormatter.purple}{MyFormatter.KW_LINENO}' \
               f'{MyFormatter.blue}, {meta_style}{meta_abv}{MyFormatter.RESET}'
        else:
            return f'[{MyFormatter.KW_NAME}] {MyFormatter.KW_FUNCNM}::{MyFormatter.KW_FNM}' \
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
    else:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        handler = logging.FileHandler(file_path)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(MyFormatter(with_color=typ == 'stdout'))
    logger.addHandler(handler)
    return logger


class RecurseLimit:
    # credit: https://stackoverflow.com/a/50120316/10732321
    def __init__(self, limit):
        self.limit = limit

    def __enter__(self):
        self.old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(self.limit)

    def __exit__(self, kind, value, tb):
        sys.setrecursionlimit(self.old_limit)

if __name__ == '__main__':
    from icecream import ic

    # ic(config('fine-tune'))

    # ic(fmt_num(124439808))

    # process_utcd_dataset()

    # map_ag_news()

    # lg = get_logger('test-lang')
    # ic(lg, type(lg))

    pass

