"""
plotting

see also `StefUtil.save_fig`
"""

from typing import Iterable

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
import matplotlib.pyplot as plt
import seaborn as sns

from stefutil.check_args import ca


__all__ = [
    'LN_KWARGS',
    'change_bar_width', 'vals2colors', 'set_color_bar', 'barplot',
]


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


LN_KWARGS = dict(marker='o', ms=0.3, lw=0.25)  # matplotlib line plot default args


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
        patch.set_x(patch.get_x() + diff * 0.5) if is_vert else patch.set_y(patch.get_y() + diff * 0.5)


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