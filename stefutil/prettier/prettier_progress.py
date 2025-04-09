import typing
import logging
from types import TracebackType
from typing import Tuple, List, Iterable, Union, Optional, Callable, Sequence
from typing import BinaryIO, ContextManager, Generic, TextIO, Type, Literal
from datetime import datetime, timedelta, timezone
from warnings import warn
from contextlib import contextmanager

from rich.progress import ProgressType, TaskProgressColumn, ProgressColumn, Progress
from tqdm.auto import tqdm
from tqdm.std import TqdmWarning
from tqdm.utils import FormatReplace, disp_len, disp_trim

from stefutil.os import rel_path
from stefutil.prettier.prettier import Timer
from stefutil.prettier.prettier_debug import style, rich_console


__all__ = [
    'rich_status', 'rich_open', 'rich_progress',
    'tqdc'
]


@contextmanager
def rich_status(desc: str = None, spinner: str = 'arrow3', transient: bool = False, logger: logging.Logger = None):
    timer = Timer() if logger else None

    status = rich_console.status(status=desc, spinner=spinner)
    if not transient:  # the rich live display's `transient` defaults to True and not exposed directly
        status._live.transient = False

    with status as status:
        yield status
    if logger:
        logger.info(f'{desc or "Task"} finished in {style(timer.end())}.')


_I = typing.TypeVar("_I", TextIO, BinaryIO)


# copied over from `rich.progress` for no public access
class _ReadContext(ContextManager[_I], Generic[_I]):
    """A utility class to handle a context for both a reader and a progress."""

    def __init__(self, progress: "Progress", reader: _I) -> None:
        self.progress = progress
        self.reader: _I = reader

    def __enter__(self) -> _I:
        self.progresstyletart()
        return self.reader.__enter__()

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.progresstyletop()
        self.reader.__exit__(exc_type, exc_val, exc_tb)


# modified from `rich.progress.open` for our custom progress bar styling
def rich_open(
    file: Union[str, "PathLike[str]", bytes],
    mode: Union[Literal["rb"], Literal["rt"], Literal["r"]] = "r"
) -> Union[ContextManager[BinaryIO], ContextManager[TextIO]]:
    desc = f'Reading file {style(rel_path(file), backend="rich-markup")}'
    progress = rich_progress(return_progress=True, desc=desc, for_file=True)

    reader = progress.open(
        file,
        mode=mode,
        description=desc,
    )
    return _ReadContext(progress, reader)


class SpeedTaskProgressColumn(TaskProgressColumn):
    """
    subclass override to always render speed like `tqdm`
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.show_speed = True

    @classmethod
    def render_speed(cls, speed: Optional[float]) -> 'rich.text.Text':
        # ======================================= Begin of added =======================================
        from rich.text import Text
        from rich import filesize
        # ======================================= End of added =======================================
        if speed is None:
            return Text("", style="progress.percentage")
        unit, suffix = filesize.pick_unit_and_suffix(
            int(speed),
            ["", "×10³", "×10⁶", "×10⁹", "×10¹²"],
            1000,
        )
        data_speed = speed / unit
        # ======================================= Begin of modified =======================================
        # return Text(f"{data_speed:.1f}{suffix} it/s", style="progress.percentage")
        return Text(f"{data_speed:.1f}{suffix}it/s", style="progress.percentage")  # drop the space
        # ======================================= End of modified =======================================

    def render(self, task: 'rich.progress.Task') -> 'rich.text.Text':
        # ======================================= Begin of modified =======================================
        # if task.total is None and self.show_speed:
        if self.show_speed:  # i.e., always show speed
            return self.render_speed(task.finished_speed or task.speed)
        # text_format = (
        #     self.text_format_no_percentage if task.total is None else self.text_format
        # )
        # _text = text_format.format(task=task)
        # if self.markup:
        #     text = Text.from_markup(_text, style=self.style, justify=self.justify)
        # else:
        #     text = Text(_text, style=self.style, justify=self.justify)
        # if self.highlighter:
        #     self.highlighter.highlight(text)
        # return text
        # ======================================= End of modified =======================================


class CompactTimeElapsedColumn(ProgressColumn):
    """
    subclass override to show time elapsed in compact format if possible
    """

    def render(self, task: 'rich.progress.Task') -> 'rich.text.Text':
        # ======================================= Begin of added =======================================
        from rich.text import Text
        from datetime import timedelta
        # ======================================= End of added =======================================
        elapsed = task.finished_time if task.finished else task.elapsed
        if elapsed is None:
            return Text("-:--:--", style="progress.elapsed")
        # ======================================= Begin of modified =======================================
        # delta = timedelta(seconds=max(0, int(elapsed)))
        # return Text(str(delta), style="progress.elapsed")
        secs = max(0, int(elapsed))
        mm, ss = divmod(secs, 60)
        hh, mm = divmod(mm, 60)
        fmt = f'{mm:02d}:{ss:02d}'
        if hh:
            fmt = f'{hh}:{fmt}'
        return Text(fmt, style="progress.elapsed")
        # ======================================= End of modified ======================================


class NoPadProgress(Progress):
    """
    subclass override to do our custom padding between progress columns
    """
    def make_tasks_table(self, tasks: Iterable['rich.progress.Task']) -> 'rich.table.Table':
        # ======================================= Begin of added =======================================
        from rich.table import Column, Table
        # ======================================= End of added =======================================
        table_columns = (
            (
                Column(no_wrap=True)
                if isinstance(_column, str)
                else _column.get_table_column().copy()
            )
            for _column in self.columns
        )

        # ======================================= Begin of modified =======================================
        # table = Table.grid(*table_columns, padding=(0, 1), expand=self.expand)
        table = Table.grid(*table_columns, padding=(0, 0), expand=self.expand)
        # ======================================= End of modified =======================================

        for task in tasks:
            if task.visible:
                table.add_row(
                    *(
                        (
                            column.format(task=task)
                            if isinstance(column, str)
                            else column(task)
                        )
                        for column in self.columns
                    )
                )
        return table


def rich_progress(
        sequence: Union[Sequence[ProgressType], Iterable[ProgressType]] = None,
        desc: Union[bool, str] = None,
        total: int = None,
        bar_width: int = None,
        return_progress: bool = False,
        fields: Union[List[str], str] = None,
        field_widths: Union[List[int], int] = None,
        for_file: bool = False,
) -> Union[Progress, Iterable[ProgressType], Tuple[Iterable[ProgressType], Callable]]:
    from rich.progress import ProgressColumn, TextColumn, BarColumn, MofNCompleteColumn, TimeRemainingColumn
    from rich.progress import DownloadColumn, TransferSpeedColumn, FileSizeColumn, TotalFileSizeColumn

    def get_pad():
        return TextColumn(' ')

    def get_semantic_pad():
        return TextColumn(' • ')

    columns: List[ProgressColumn] = []
    if desc:
        columns.append(TextColumn('[progress.description]{task.description}'))

    pbar = BarColumn(bar_width=bar_width)
    columns += [
        get_pad(), pbar,
        get_pad(), TaskProgressColumn(), get_pad()
    ]
    if for_file:
        columns += [FileSizeColumn(), TextColumn('/'), TotalFileSizeColumn()]
        # DownloadColumn(),
    else:
        columns.append(MofNCompleteColumn())
    columns += [
        get_semantic_pad(), CompactTimeElapsedColumn(), TextColumn('>'), TimeRemainingColumn(compact=True),
        get_pad(), TransferSpeedColumn() if for_file else SpeedTaskProgressColumn()
    ]

    if fields:
        if isinstance(fields, str):
            fields = [fields]

        if field_widths:
            if isinstance(field_widths, int):
                field_widths = [field_widths] * len(fields)
            elif len(field_widths) != len(fields):
                raise ValueError(f'Length of {style("field_widths")} must match {style("fields")}')
        else:
            field_widths = [4] * len(fields)

    has_fields = fields and len(fields) > 0
    if has_fields:
        n = len(fields)
        columns.append(get_semantic_pad())

        if n > 1:  # add enclosing braces before & after
            columns.append(TextColumn('[magenta]{{'))
        for i, (key, w) in enumerate(zip(fields, field_widths)):
            columns += [TextColumn(f"{key}=[blue]{{task.fields[{key}]:>{w}}}")]  # align to right

            if i < n - 1:
                columns.append(TextColumn(', '))
        if n > 1:
            columns.append(TextColumn('[magenta]}}'))

    progress = NoPadProgress(*columns)
    if return_progress:
        return progress

    else:
        if not has_fields:
            def ret():
                with progress:
                    yield from progress.track(sequence, total=total, description=desc)
            return ret()
        else:
            from rich.progress import length_hint, _TrackThread
            # modified from `rich.progress.track`
            if total is None:
                total = float(length_hint(sequence)) or None

            task_args = {k: '_' * w for k, w in zip(fields, field_widths)} if fields else dict()
            task_id = progress.add_task(desc, total=total, **task_args)

            assert progress.live.auto_refresh

            def update_callback(**fields_):
                if not progress.finished:
                    # progress.update(task_id, advance=1, **fields_)
                    progress.update(task_id, advance=0, **fields_)

            def _ret():
                with _TrackThread(progress=progress, task_id=task_id, update_period=0.1) as track_thread:
                    for value in sequence:
                        yield value
                        track_thread.completed += 1

            def ret():
                with progress:
                    yield from _ret()
            return ret(), update_callback


# copied over from `tqdm.std` for it's not publicly visible; override to use our coloring
class CBar(object):
    ASCII = " 123456789#"
    UTF = u" " + u''.join(map(chr, range(0x258F, 0x2587, -1)))
    BLANK = "  "
    COLOUR_RESET = '\x1b[0m'
    COLOUR_RGB = '\x1b[38;2;%d;%d;%dm'
    COLOURS = {'BLACK': '\x1b[30m', 'RED': '\x1b[31m', 'GREEN': '\x1b[32m',
               'YELLOW': '\x1b[33m', 'BLUE': '\x1b[34m', 'MAGENTA': '\x1b[35m',
               'CYAN': '\x1b[36m', 'WHITE': '\x1b[37m'}

    def __init__(self, frac, default_len=10, charset=UTF, colour=None):
        if not 0 <= frac <= 1:
            warn("clamping frac to range [0, 1]", TqdmWarning, stacklevel=2)
            frac = max(0, min(1, frac))
        assert default_len > 0
        self.frac = frac
        self.default_len = default_len
        self.charset = charset
        self.colour = colour

    @property
    def colour(self):
        return self._colour

    @colour.setter
    def colour(self, value):
        # ======================================= Begin of modified =======================================
        # if not value:
        #     self._colour = None
        #     return
        # try:
        #     if value.upper() in self.COLOURS:
        #         self._colour = self.COLOURS[value.upper()]
        #     elif value[0] == '#' and len(value) == 7:
        #         self._colour = self.COLOUR_RGB % tuple(
        #             int(i, 16) for i in (value[1:3], value[3:5], value[5:7]))
        #         sic('di you go here')
        #     else:
        #         raise KeyError
        # except (KeyError, AttributeError):
        #     warn("Unknown colour (%s); valid choices: [hex (#00ff00), %s]" % (
        #          value, ", ".join(self.COLOURS)),
        #          TqdmWarning, stacklevel=2)
        #     self._colour = None
        self._colour = value
        # ======================================= End of modified =======================================

    def __format__(self, format_spec):
        if format_spec:
            _type = format_spec[-1].lower()
            try:
                charset = {'a': self.ASCII, 'u': self.UTF, 'b': self.BLANK}[_type]
            except KeyError:
                charset = self.charset
            else:
                format_spec = format_spec[:-1]
            if format_spec:
                N_BARS = int(format_spec)
                if N_BARS < 0:
                    N_BARS += self.default_len
            else:
                N_BARS = self.default_len
        else:
            charset = self.charset
            N_BARS = self.default_len

        nsyms = len(charset) - 1
        bar_length, frac_bar_length = divmod(int(self.frac * N_BARS * nsyms), nsyms)

        res = charset[-1] * bar_length
        if bar_length < N_BARS:  # whitespace padding
            # ======================================= Begin of modified =======================================
            # res = res + charset[frac_bar_length] + charset[0] * (N_BARS - bar_length - 1)
            # instead of empty space, use a grey dashed line; style is just like the default for `rich.progress.Progress`
            if charset[0] != ' ':
                raise NotImplementedError
            res = res + charset[frac_bar_length] + style.nb('━' * (N_BARS - bar_length - 1), fg='grey23')
            # ======================================= End of modified =======================================
        # ======================================= Begin of modified =======================================
        # return self.colour + res + self.COLOUR_RESET if self.colour else res
        return style(res, fg=self.colour, bold=False) if self.colour else res
        # ======================================= End of modified =======================================


# copied over from `tqdm.utils` for not publicly visible
def _is_utf(encoding):
    try:
        u'\u2588\u2589'.encode(encoding)
    except UnicodeEncodeError:
        return False
    except Exception:
        try:
            return encoding.lower().startswith('utf-') or ('U8' == encoding)
        except Exception:
            return False
    else:
        return True


def _supports_unicode(fp):
    try:
        return _is_utf(fp.encoding)
    except AttributeError:
        return False


def _is_ascii(s):
    if isinstance(s, str):
        for c in s:
            if ord(c) > 255:
                return False
        return True
    return _supports_unicode(s)


def _style_interval(x: int, pad: bool = False) -> str:
    return style(x, bold=False, fg='g', pad='02' if pad else None)


def _style_rate(x: float) -> str:
    return style(x, bold=False, fg='c', pad='4.2f')


class tqdc(tqdm):
    """
    override tqdm for custom coloring
    """
    def __init__(self, iterable=None, colour: str = 'red', ascii: str = ' ╺━', **kwargs):  # change default color to show that task not completed
        super().__init__(iterable, colour=colour, ascii=ascii, **kwargs)
        # if self.total is None:
        #     from operator import length_hint
        #     self.total = length_hint(iterable)

    @staticmethod
    def format_interval(t):
        # ======================================= Begin of modified =======================================
        mm, ss = divmod(int(t), 60)
        hh, mm = divmod(mm, 60)
        return f'{_style_interval(hh)}:{_style_interval(mm, pad=True)}:{_style_interval(ss, pad=True)}' if hh else f'{_style_interval(mm, pad=True)}:{_style_interval(ss, pad=True)}'
        # ======================================= End of modified =======================================

    @staticmethod
    def format_meter(n, total, elapsed, ncols=None, prefix='', ascii=False, unit='it',
                     unit_scale=False, rate=None, bar_format=None, postfix=None,
                     unit_divisor=1000, initial=0, colour=None, **extra_kwargs):
        # sanity check: total
        if total and n >= (total + 0.5):  # allow float imprecision (#849)
            total = None

        # apply custom scale if necessary
        if unit_scale and unit_scale not in (True, 1):
            if total:
                total *= unit_scale
            n *= unit_scale
            if rate:
                rate *= unit_scale  # by default rate = self.avg_dn / self.avg_dt
            unit_scale = False

        # ======================================= Begin of modified =======================================
        # elapsed_str = tqdm.format_interval(elapsed)
        elapsed_str = tqdc.format_interval(elapsed)
        # ======================================= End of modified =======================================

        # if unspecified, attempt to use rate = average speed
        # (we allow manual override since predicting time is an arcane art)
        if rate is None and elapsed:
            rate = (n - initial) / elapsed
        inv_rate = 1 / rate if rate else None
        format_sizeof = tqdm.format_sizeof
        # ======================================= Begin of modified =======================================
        # rate_noinv_fmt = ((format_sizeof(rate) if unit_scale else f'{rate:5.2f}')
        #                   if rate else '?') + unit + '/s'
        # rate_inv_fmt = (
        #     (format_sizeof(inv_rate) if unit_scale else f'{inv_rate:5.2f}')
        #     if inv_rate else '?') + 's/' + unit
        rate_noinv_fmt = ((format_sizeof(rate, divisor=unit_divisor) if unit_scale else _style_rate(rate)) if rate else '?') + unit + '/s'
        rate_inv_fmt = ((format_sizeof(inv_rate, divisor=unit_divisor) if unit_scale else _style_rate(inv_rate)) if inv_rate else '?') + 's/' + unit
        # ======================================= End of modified =======================================
        rate_fmt = rate_inv_fmt if inv_rate and inv_rate > 1 else rate_noinv_fmt

        if unit_scale:
            n_fmt = format_sizeof(n, divisor=unit_divisor)
            total_fmt = format_sizeof(total, divisor=unit_divisor) if total is not None else '?'
        else:
            n_fmt = str(n)
            total_fmt = str(total) if total is not None else '?'
        # ======================================= Begin of added =======================================
        n_fmt = style(n_fmt, bold=False, fg='c')
        total_fmt = style(total_fmt, bold=False, fg='c')
        # ======================================= End of added =======================================

        try:
            postfix = ', ' + postfix if postfix else ''
        except TypeError:
            pass

        remaining = (total - n) / rate if rate and total else 0
        # ======================================= Begin of modified =======================================
        # remaining_str = tqdm.format_interval(remaining) if rate else '?'
        remaining_str = tqdc.format_interval(remaining) if rate else '?'
        # ======================================= End of modified =======================================
        try:
            eta_dt = (datetime.now() + timedelta(seconds=remaining)
                      if rate and total else datetime.fromtimestamp(0, timezone.utc))
        except OverflowError:
            eta_dt = datetime.max

        # format the stats displayed to the left and right sides of the bar
        if prefix:
            # old prefix setup work around
            bool_prefix_colon_already = (prefix[-2:] == ": ")
            l_bar = prefix if bool_prefix_colon_already else prefix + ": "
        else:
            l_bar = ''

        r_bar = f'| {n_fmt}/{total_fmt} [{elapsed_str}<{remaining_str}, {rate_fmt}{postfix}]'

        # Custom bar formatting
        # Populate a dict with all available progress indicators
        format_dict = {
            # slight extension of self.format_dict
            'n': n, 'n_fmt': n_fmt, 'total': total, 'total_fmt': total_fmt,
            'elapsed': elapsed_str, 'elapsed_s': elapsed,
            'ncols': ncols, 'desc': prefix or '', 'unit': unit,
            'rate': inv_rate if inv_rate and inv_rate > 1 else rate,
            'rate_fmt': rate_fmt, 'rate_noinv': rate,
            'rate_noinv_fmt': rate_noinv_fmt, 'rate_inv': inv_rate,
            'rate_inv_fmt': rate_inv_fmt,
            'postfix': postfix, 'unit_divisor': unit_divisor,
            'colour': colour,
            # plus more useful definitions
            'remaining': remaining_str, 'remaining_s': remaining,
            'l_bar': l_bar, 'r_bar': r_bar, 'eta': eta_dt,
            **extra_kwargs}

        # total is known: we can predict some stats
        if total:
            # fractional and percentage progress
            frac = n / total
            percentage = frac * 100

            l_bar += f'{percentage:3.0f}%|'

            if ncols == 0:
                return l_bar[:-1] + r_bar[1:]

            format_dict.update(l_bar=l_bar)
            if bar_format:
                format_dict.update(percentage=percentage)

                # auto-remove colon for empty `{desc}`
                if not prefix:
                    bar_format = bar_format.replace("{desc}: ", '')
            else:
                bar_format = "{l_bar}{bar}{r_bar}"

            full_bar = FormatReplace()
            nobar = bar_format.format(bar=full_bar, **format_dict)
            if not full_bar.format_called:
                return nobar  # no `{bar}`; nothing else to do

            # Formatting progress bar space available for bar's display
            # ======================================= Begin of modified =======================================
            # full_bar = Bar(frac,
            #                max(1, ncols - disp_len(nobar)) if ncols else 10,
            #                charset=Bar.ASCII if ascii is True else ascii or Bar.UTF,
            #                colour=colour)
            full_bar = CBar(
                frac=frac, default_len=max(1, ncols - disp_len(nobar)) if ncols else 10,
                charset=CBar.ASCII if ascii is True else ascii or CBar.UTF, colour=colour)
            # ======================================= End of modified =======================================
            if not _is_ascii(full_bar.charset) and _is_ascii(bar_format):
                bar_format = str(bar_format)
            res = bar_format.format(bar=full_bar, **format_dict)
            return disp_trim(res, ncols) if ncols else res

        elif bar_format:
            # user-specified bar_format but no total
            l_bar += '|'
            format_dict.update(l_bar=l_bar, percentage=0)
            full_bar = FormatReplace()
            nobar = bar_format.format(bar=full_bar, **format_dict)
            if not full_bar.format_called:
                return nobar
            # ======================================= Begin of modified =======================================
            # full_bar = Bar(0,
            #                max(1, ncols - disp_len(nobar)) if ncols else 10,
            #                charset=Bar.BLANK, colour=colour)
            full_bar = CBar(frac=0, default_len=max(1, ncols - disp_len(nobar)) if ncols else 10, charset=CBar.BLANK, colour=colour)
            # ======================================= End of modified =======================================
            res = bar_format.format(bar=full_bar, **format_dict)
            return disp_trim(res, ncols) if ncols else res
        else:
            # no total: no progressbar, ETA, just progress stats
            return (f'{(prefix + ": ") if prefix else ""}'
                    f'{n_fmt}{unit} [{elapsed_str}, {rate_fmt}{postfix}]')

    def __iter__(self):
        """Backward-compatibility to use: for x in tqdm(iterable)"""

        # Inlining instance variables as locals (speed optimisation)
        iterable = self.iterable

        # If the bar is disabled, then just walk the iterable
        # (note: keep this check outside the loop for performance)
        if self.disable:
            for obj in iterable:
                yield obj
            return

        mininterval = self.mininterval
        last_print_t = self.last_print_t
        last_print_n = self.last_print_n
        min_start_t = self.start_t + self.delay
        n = self.n
        time = self._time

        try:
            for obj in iterable:
                yield obj
                # Update and possibly print the progressbar.
                # Note: does not call self.update(1) for speed optimisation.
                n += 1

                if n - last_print_n >= self.miniters:
                    cur_t = time()
                    dt = cur_t - last_print_t
                    if dt >= mininterval and cur_t >= min_start_t:
                        self.update(n - last_print_n)
                        last_print_n = self.last_print_n
                        last_print_t = self.last_print_t
        finally:
            self.n = n
            # ======================================= Begin of added =======================================
            if self.n == self.total:  # completed => change color to show so
                self.colour = 'green'
                self.update(0)
            # ======================================= End of added =======================================
            self.close()


if __name__ == '__main__':
    from rich.traceback import install
    install()

    from stefutil.prettier.prettier_debug import sic

    def check_rich_pbar():
        import time
        for i in rich_progress(range(100), desc='Processing...'):
            time.sleep(0.05)
    # check_rich_pbar()

    # def check_rich_pbar_prog():
    #     import time
    #     import random
    #
    #     with rich_progress(desc='Processing...', fields='dur') as progress:
    #         task_id = progress.add_task('blah', total=1000, dur='--')
    #         while not progress.finished:
    #             t_ms = random.randint(5, 500)
    #             progress.update(task_id, advance=1, dur=t_ms)
    #             time.sleep(t_ms / 1000)
    # check_rich_pbar_prog()

    def check_rich_pbar_field():
        import time
        import random

        # seq = range(100)
        seq = range(20)
        # desc = f'Processing {s.i("hey")}...'  # TODO: try their styling
        desc = f'Processing [bold green]hey[/bold green]...'
        it, update = rich_progress(sequence=seq, desc=desc, fields=['dur', 'char'])
        for i in it:
            t_ms = random.randint(5, 500)
            ch = random.sample('abcde', 2)
            ch = ''.join(ch)
            # print(ch)
            # raise NotImplementedError
            # sic(ch)
            update(dur=t_ms, char=ch)
            time.sleep(t_ms / 1000)
    # check_rich_pbar_field()

    def check_rich_backend_colors():
        txt = 'hello'
        for c in ['magenta', 'dodger_blue2', 'dark_red']:
            print(c + style(txt, fg=c))
    # check_rich_backend_colors()

    def check_nested_rich_pbar():
        import time
        import random

        progress = rich_progress(return_progress=True, desc=True)
        with progress:
            for i in progress.track(range(20), description='outer'):
                for j in progress.track(range(5), description=f'inner {i}'):
                    t_ms = random.randint(5, 300)
                    time.sleep(t_ms / 1000)
    # check_nested_rich_pbar()

    def check_tqdm_color():
        import time
        import random
        from tqdm.auto import tqdm

        n = 40
        # n = 4000
        for i in tqdc(range(n), desc='Processing...'):
            t_ms = random.randint(5, 300)
            # t_ms = random.randint(5, 10)
            time.sleep(t_ms / 1000)
    # check_tqdm_color()

    def check_rich_open():
        import rich.progress
        path = '../../stefutil/../stefutil/test-both-handler.log.ansi'
        # with rich.progress.open(path, 'r') as f:
        with rich_open(path) as f:
            txt = f.read()
        sic(txt)
    # check_rich_open()

    def check_rich_open_large():
        def write_large():
            # write a large file containing random 10M characters
            import random
            import string
            path = 'large.txt'
            with open(path, 'w') as f:
                for _ in tqdc(range(100_000_000)):
                    f.write(random.choice(string.ascii_letters))
        write_large()

        with rich_open('large.txt') as f:
            # read 10K characters at a time
            for _ in range(10_000):
                txt = f.read(10_000)
                import time
                time.sleep(0.001)
        sic(txt[:10])
    # check_rich_open_large()

    def check_rich_status():
        import time
        import random

        from stefutil.prettier.prettier_log import get_logger

        _logger = get_logger('test')
        _logger.info(fr'ya')
        with rich_status(desc='task rand', logger=_logger):
            time.sleep(random.random() * 4)
    check_rich_status()
