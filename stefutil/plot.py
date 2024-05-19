"""
plotting

see also `StefUtil.save_fig`
"""


import math
import logging
from typing import List, Dict, Iterable, Callable, Any, Union
from dataclasses import dataclass

from stefutil.prettier import s, ca, get_logger, Timer
from stefutil.container import df_col2cat_col
from stefutil.packaging import _use_plot, _use_ml


__all__ = []


_logger = get_logger(__name__)


if _use_plot():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.patches import Ellipse
    from matplotlib.colors import to_rgba
    import matplotlib.transforms as transforms

    __all__ += [
        'set_plot_style', 'LN_KWARGS',
        'change_bar_width', 'vals2colors', 'set_color_bar', 'barplot'
    ]


    def set_plot_style():
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
        ca(bar_orient=orient)
        is_vert = orient in ['v', 'vertical']
        for patch in ax.patches:
            current_width = patch.get_width() if is_vert else patch.get_height()
            diff = current_width - width
            patch.set_width(width) if is_vert else patch.set_height(width)
            patch.set_x(patch.get_x() + diff * 0.5) if is_vert else patch.set_y(patch.get_y() + diff * 0.5)


    def vals2colors(
            vals: Iterable[float], color_palette: str = 'Spectral_r', gap: float = None,
    ) -> List:
        """
        Map an iterable of values to corresponding colors given a color map

        :param vals: Values to map color
        :param color_palette: seaborn color map
        :param gap: Ratio of difference between min and max values
            If given, reduce visual spread/difference of colors
            Intended for a less drastic color at the extremes
        """
        vals = np.asarray(vals)
        cmap = sns.color_palette(color_palette, as_cmap=True)
        mi, ma = np.min(vals), np.max(vals)
        if gap is not None:
            diff = ma - mi
            mi, ma = mi - diff * gap, ma + diff * gap
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
            data: pd.DataFrame = None,
            x: Union[Iterable, str] = None, y: Union[Iterable[float], str] = None,
            x_order: Iterable[str] = None,
            orient: str = 'v', with_value: bool = False, width: [float, bool] = 0.5,
            xlabel: str = None, ylabel: str = None, yscale: str = None, title: str = None,
            ax=None, palette: Union[str, List, Any] = 'husl', callback: Callable[[plt.Axes], None] = None,
            show: bool = True,
            **kwargs
    ):
        ca(bar_orient=orient)
        if data is not None:
            df = data
            assert isinstance(x, str) and isinstance(y, str)
            df['x'], df['y'] = df[x], df[y]
        else:
            df = pd.DataFrame([dict(x=x_, y=y_) for x_, y_ in zip(x, y)])
            x_order = list(x)
        if x_order is not None:
            x_order: List[str]
            df_col2cat_col(df, 'x', categories=x_order)
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
        if yscale:
            ax.set_yscale(yscale) if is_vert else ax.set_xscale(yscale)
        if title:
            ax.set_title(title)
        if callback:
            callback(ax)
        if show:
            plt.show()
        return ax


    def confidence_ellipse(ax_, x, y, n_std=1., **kws):
        """
        Modified from https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html
        Create a plot of the covariance confidence ellipse of x and y

        :param ax_: matplotlib axes object to plot ellipse on
        :param x: x values
        :param y: y values
        :param n_std: number of standard deviations to determine the ellipse's radius'
        :return matplotlib.patches.Ellipse
        """
        cov = np.cov(x, y)
        pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
        r_x, r_y = np.sqrt(1 + pearson), np.sqrt(1 - pearson)
        _args = {**dict(fc='none'), **kws}
        ellipse = Ellipse((0, 0), width=r_x * 2, height=r_y * 2, **_args)
        scl_x, scl_y = np.sqrt(cov[0, 0]) * n_std, np.sqrt(cov[1, 1]) * n_std
        mu_x, mu_y = np.mean(x), np.mean(y)
        tsf = transforms.Affine2D().rotate_deg(45).scale(scl_x, scl_y).translate(mu_x, mu_y)
        ellipse.set_transform(tsf + ax_.transData)
        return ax_.add_patch(ellipse)

    if _use_ml():
        __all__ += ['VecProjOutput', 'vector_projection_plot']

        @dataclass
        class VecProjOutput:
            df: pd.DataFrame = None
            ax: plt.Axes = None


        def vector_projection_plot(
                name2vectors: Dict[str, np.ndarray], tsne_args: Dict[str, Any] = None, tight_fig_size: bool = True, key_name: str = 'setup',
                ellipse: bool = True, ellipse_std: Union[int, float] = 1,
                scatter_ms: Union[int, float] = 48, scatter_kwargs: Dict[str, Any] = None, ax: plt.Axes = None,
                title: str = None, verbose: bool = True, logger: logging.Logger = None
        ):
            """
            Given vectors grouped by key, plot projections of vectors into 2D space
                Intended for plotting embedding space of SBert sentence representations

            :param name2vectors: 2D vectors grouped by setup name
            :param tsne_args: Arguments for TSNE dimensionality reduction
            :param tight_fig_size: If true, resize the figure to fit the axis range
            :param key_name: column name for setup in the internal dataframe
            :param ellipse: If true, plot confidence ellipse for each setup
            :param ellipse_std: Number of standard deviations for ellipse
            :param scatter_ms: Base marker size for scatter plot
                Will be scaled by number of samples in each group
            :param scatter_kwargs: arguments for scatter plot
            :param ax: matplotlib axes object to plot on
            :param title: plot title
            :param verbose: If true, prints status to logger
            :param logger: logger
            """
            from sklearn.manifold import TSNE  # lazy import to save time
            vects = np.concatenate(list(name2vectors.values()), axis=0)
            tsne_args_ = dict(n_components=2, perplexity=50, random_state=42)
            tsne_args_.update(tsne_args or dict())

            logger = logger or _logger
            if verbose:
                logger.info(f'Running TSNE on {s.i(len(vects))} vectors w/ args {s.i(tsne_args_)}')
            t = Timer()
            vects_reduced = TSNE(**tsne_args_).fit_transform(vects)
            if verbose:
                logger.info(f'TSNE finished in {s.i(t.end())}')

            setups = sum([[nm] * len(v) for nm, v in name2vectors.items()], start=[])
            df = pd.DataFrame({'x': vects_reduced[:, 0], 'y': vects_reduced[:, 1], key_name: setups})

            ms = 48 or scatter_ms  # more samples => smaller markers
            dnm2ms = {nm: 1 / math.log((len(v))) * ms for nm, v in name2vectors.items()}
            cs = sns.color_palette('husl', n_colors=len(name2vectors))
            sct_args = dict(hue=key_name, palette=cs, size=key_name, sizes=dnm2ms, alpha=0.7, ax=ax)
            sct_args.update(scatter_kwargs or dict())

            if verbose:
                logger.info(f'Plotting embedded 2D points w/ args {s.i(sct_args, indent=1)}')
            t = Timer()
            ax = sns.scatterplot(data=df, x='x', y='y', **sct_args)
            if verbose:
                logger.info(f'Scatterplot finished in {s.i(t.end())}')

            if ellipse:
                for nm, c in zip(name2vectors.keys(), cs):
                    x, y = df[df[key_name] == nm]['x'].values, df[df[key_name] == nm]['y'].values
                    confidence_ellipse(ax_=ax, x=x, y=y, n_std=ellipse_std, fc=to_rgba(c, 0.1), ec=to_rgba(c, 0.6))

            ax.set_aspect('equal')
            if tight_fig_size:
                # resize the figure w.r.t axis range
                (x_min, x_max), (y_min, y_max) = ax.get_xlim(), ax.get_ylim()
                ratio = (x_max - x_min) / (y_max - y_min)
                base_area = 100
                height, weight = math.sqrt(base_area / ratio), math.sqrt(base_area * ratio)
                plt.gcf().set_size_inches(weight, height)

            if title:
                plt.suptitle(title)
            return VecProjOutput(df=df, ax=ax)
