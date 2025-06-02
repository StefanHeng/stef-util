"""
plotting

see also `StefUtil.save_fig`
"""


import math
import logging
from typing import List, Dict, Iterable, Callable, Any, Union
from dataclasses import dataclass

from pydantic import BaseModel

from stefutil.prettier import style as s, check_arg as ca, get_logger, Timer
from stefutil.container import df_col2cat_col
from stefutil.packaging import installed_packages, _use_plot, _use_ml


__all__ = []


_logger = get_logger(__name__)


if _use_plot():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    from matplotlib.colors import to_rgba
    import matplotlib.colors as colors
    import matplotlib.transforms as transforms

    __all__ += [
        'set_plot_style', 'LN_KWARGS',
        'change_bar_width', 'vals2colors', 'truncate_colormap', 'set_color_bar', 'barplot'
    ]


    def set_plot_style():
        plt.rich_console('figure', figsize=(16, 9))
        plt.rich_console('figure.constrained_layout', use=True)
        plt.rich_console('text.latex', preamble='\n'.join([
            r'\usepackage{nicefrac}',
            r'\usepackage{helvet}',
            r'\usepackage{sansmath}',  # render math sans-serif
            r'\sansmath'
        ]))
        if 'seaborn' in installed_packages():
            import seaborn as sns
            snstyleet_style('darkgrid')
            snstyleet_context(rc={'grid.linewidth': 0.5})

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
        import seaborn as sns
        vals = np.asarray(vals)
        cmap = sns.color_palette(color_palette, as_cmap=True)
        mi, ma = np.min(vals), np.max(vals)
        if gap is not None:
            diff = ma - mi
            mi, ma = mi - diff * gap, ma + diff * gap
        norm = (vals - mi) / (ma - mi)
        return cmap(norm)

    def truncate_colormap(cmap, vmin=0.0, vmax=1.0, n=100):
        if isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)
        cmap: colors.Colormap
        new_cmap = colors.LinearSegmentedColormap.from_list(
            name='trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=vmin, b=vmax), colors=cmap(np.linspace(vmin, vmax, n)))
        return new_cmap

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
        import seaborn as sns

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


    class AnalyzeDistOutput(BaseModel):
        top_n_values: dict[int, float] | None = None
        percentile_cutoffs: dict[float, float] | None = None


    def analyze_n_plot_dist(
            values: 'np.array' = None, save_path: str = None, title: str = None, xlabel: str = None, plot: bool = True,
            top_n: int = None, percentiles: list[float] = None,
            additional_values: dict[str, 'np.array'] = None,
            logger: logging.Logger = _logger, percentile_log_decimal: int = 2
    ) -> AnalyzeDistOutput:
        """
        Plot the distribution of some values side by side
            left: histogram, right: cumulative histogram
        """
        from os.path import join as os_join

        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns

        from stefutil.prettier import style, round_f, now

        vals = np.asarray(values)

        idx2val, p2c = None, None
        if top_n:  # log top largest values along w/ the indices
            idxs = np.argsort(vals)[-top_n:]
            idx2val = dict(zip(idxs, vals[idxs]))
            logger.info(f'{xlabel.capitalize()} top {style(top_n)} largest values: {style(idx2val, indent=1)}')

        if percentiles:
            # get the percentile cutoffs
            assert all(0 <= p <= 1 for p in percentiles)  # sanity check
            percentiles = sorted([p * 100 for p in percentiles])

            cutoffs = np.percentile(vals, percentiles)
            p2c = dict(zip(percentiles, cutoffs))
            logger.info(f'{xlabel.capitalize()} percentile cutoffs: {style({p: round_f(c, decimal=percentile_log_decimal) for p, c in p2c.items()}, indent=1)}')

        if plot:
            # default on left, a cumulative one on the right
            fig, axs = plt.subplots(1, 2, figsize=(16, 9))

            has_more_val = additional_values and len(additional_values)

            args_left = dict(stat='percent', kde=True, ax=axs[0])
            args_right = dict(stat='percent', cumulative=True, fill=False, element='step', ax=axs[1])

            if has_more_val:
                import pandas as pd

                additional_values = {k: np.asarray(v) for k, v in additional_values.items()}
                # plot each of them independently by `hue`
                rows = [dict(kind=xlabel, value=val) for val in vals]
                for k, v in additional_values.items():
                    rows.extend([dict(kind=k, value=val) for val in v])
                df = pd.DataFrame(rows)
                sns.histplot(data=df, x='value', hue='kind', **args_left)
                sns.histplot(data=df, x='value', hue='kind', **args_right)

            else:
                sns.histplot(vals, **args_left)
                sns.histplot(vals, **args_right)

            if percentiles:
                # add vertical lines for the percentile cutoffs to the cumulative plot
                cs = sns.color_palette(palette='husl', n_colors=len(p2c))
                for color, (p, c) in zip(cs, p2c.items()):
                    axs[1].axvline(c, color=color, linestyle='--', label=f'{p}% cutoff', linewidth=0.5)

            if has_more_val or percentiles:
                plt.legend()

            title = title or 'Dist. of values'
            xlab = xlabel or 'value'
            plt.suptitle(title)
            fig.supxlabel(f'{xlab}')
            fig.supylabel(f'% of {xlab}')
            plt.tight_layout()

            if save_path:
                time = now(fmt='short-full', for_path=True)
                title = title.replace(' ', '-')
                fnm = f'{time}_{title}.png'
                path = os_join(save_path, fnm)
                logger.info(f'Saving distribution plot to {style(path)}...')
                plt.savefig(path, dpi=300)

        return AnalyzeDistOutput(top_n_values=idx2val, percentile_cutoffs=p2c)

    if _use_ml():
        __all__ += ['VecProjOutput', 'vector_projection_plot']

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

            :param name2vectors: 2D vectors grouped by setup name.
            :param tsne_args: Arguments for TSNE dimensionality reduction.
            :param tight_fig_size: If true, resize the figure to fit the axis range.
            :param key_name: column name for setup in the internal dataframe.
            :param ellipse: If true, plot confidence ellipse for each setup.
            :param ellipse_std: Number of standard deviations for ellipse.
            :param scatter_ms: Base marker size for scatter plot.
                Will be scaled by number of samples in each group.
            :param scatter_kwargs: arguments for scatter plot.
            :param ax: matplotlib axes object to plot on.
            :param title: plot title.
            :param verbose: If true, prints status to logger.
            :param logger: logger.
            """
            from sklearn.manifold import TSNE  # lazy import to save time
            import seaborn as sns

            vects = np.concatenate(list(name2vectors.values()), axis=0)
            tsne_args_ = dict(n_components=2, perplexity=50, random_state=42)
            tsne_args_.update(tsne_args or dict())

            logger = logger or _logger
            if verbose:
                logger.info(f'Running TSNE on {style(len(vects))} vectors w/ args {style(tsne_args_)}')
            t = Timer()
            vects_reduced = TSNE(**tsne_args_).fit_transform(vects)
            if verbose:
                logger.info(f'TSNE finished in {style(t.end())}')

            setups = sum([[nm] * len(v) for nm, v in name2vectors.items()], start=[])
            df = pd.DataFrame({'x': vects_reduced[:, 0], 'y': vects_reduced[:, 1], key_name: setups})

            ms = 48 or scatter_ms  # more samples => smaller markers
            dnm2ms = {nm: 1 / math.log((len(v))) * ms for nm, v in name2vectors.items()}
            cs = sns.color_palette('husl', n_colors=len(name2vectors))
            sct_args = dict(hue=key_name, palette=cs, size=key_name, sizes=dnm2ms, alpha=0.7, ax=ax)
            sct_args.update(scatter_kwargs or dict())

            if verbose:
                logger.info(f'Plotting embedded 2D points w/ args {style(sct_args, indent=1)}')
            t = Timer()
            ax = snstylecatterplot(data=df, x='x', y='y', **sct_args)
            if verbose:
                logger.info(f'Scatterplot finished in {style(t.end())}')

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
