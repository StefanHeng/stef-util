"""
Project & project file structure related
"""

import os
import json
from os.path import join as os_join
from typing import List, Dict, Union

from stefutil.container import get
from stefutil.os_n_file import rel_path
from stefutil.prettier import now, ca, get_logger, s


__all__ = ['SConfig', 'PathUtil']


logger = get_logger(__name__)


class SConfig:
    """
    the one-stop place for package-level constants, expects a json file
    """
    def __init__(self, config_file: str):
        self.config_file = config_file
        with open(config_file, 'r') as f:
            self.d = json.load(f)

    def __call__(self, keys: str = None):
        """
        Retrieves the queried attribute value from the config file

        Loads the config file on first call.
        """
        return get(dic=self.d, ks=keys)


class PathUtil:
    """
    Effectively curried functions with my enforced project & dataset structure
        Pass in file paths
    """
    plot_dir = 'plot'
    eval_dir = 'eval'

    def __init__(
            self, base_path: str = None, project_dir: str = None, package_name: str = None,
            dataset_dir: str = None, model_dir: str = None, within_proj: bool = True, makedirs: Union[bool, str, List[str]] = True,
            verbose: bool = True
    ):
        """
        :param base_path: Absolute system path for root directory that contains a project folder & a data folder
        :param project_dir: Project root directory name that contains a folder for main source files
        :param package_name: python package/Module name which contain main source files
        :param dataset_dir: Directory name that contains datasets
        :param model_dir: Directory name that contains trained models
        :param within_proj: If true, model and dataset directories are under project directory
        :param makedirs: If true, create directories if they don't exist
            If a list is passed in, create only those directories
        :param verbose: If true, log messages are printed
        """
        self.base_path = base_path
        self.proj_dir = project_dir
        self.pkg_nm = package_name
        self.dset_dir = dataset_dir
        self.model_dir = model_dir
        self.within_proj = within_proj

        self.proj_path = os_join(self.base_path, self.proj_dir)
        base_path = self.proj_path if within_proj else self.base_path
        self.dset_path = os_join(base_path, self.dset_dir)
        self.model_path = os_join(base_path, self.model_dir)

        self.plot_path = os_join(self.base_path, self.proj_dir, PathUtil.plot_dir)
        self.eval_path = os_join(self.base_path, self.proj_dir, PathUtil.eval_dir)

        self.makedirs = makedirs
        self.verbose = verbose

        if makedirs:
            dirs_list = ['dataset', 'model', 'plot', 'eval']
            if makedirs is True:
                dirs = dirs_list
            elif isinstance(makedirs, str):
                dirs = [makedirs]
            else:
                assert isinstance(makedirs, list)
                dirs = makedirs
            dir2path = dict(dataset=self.dset_path, model=self.model_path, plot=self.plot_path, eval=self.eval_path)
            for dir_ in dirs:
                ca.assert_options(display_name='Directories to create', val=dir_, options=dirs_list)
                path = dir2path[dir_]
                if not os.path.exists(path):
                    os.makedirs(path)
                    if verbose:
                        logger.info(f'Created directory {s.i(path)}')

    def save_fig(
            self, title: str = None, save: bool = True, prefix_time: bool = True, save_path: str = None, time_args: Dict = None,
            fmt: str = 'png', **kwargs
    ):
        """
        :param title: Rendered figure title.
        :param save: If true, the figure is saved to project plot directory.
            No effect otherwise
        :param prefix_time: If true, timestamp is prefixed before filename.
            Otherwise, timestamp is appended to the end.
        :param save_path: Disk path to save the figure.
        :param time_args: `now` arguments/
        :param fmt: file format.
        """
        if save:
            try:
                import matplotlib.pyplot as plt
            except ImportError:
                raise ImportError('`matplotlib` not found in the environment. Please install the package.')

            args = dict(fmt='short-full', for_path=True)
            if time_args is not None:  # for python3.8 compatibility
                args.update(time_args)
            t = now(**args)

            if title is None:
                title = 'Figure'
            if 'w/' in title:
                title = title.replace('w/', 'with')
            elif '/' in title:
                raise ValueError(f'Invalid title {s.i(title)} for containing [{s.i("/")}]')

            if prefix_time:
                fnm = f'{t}_{title}'
            else:
                fnm = f'{title}, {t}'
            fnm = f'{fnm}.{fmt}'
            path = os_join((save_path or self.plot_path), fnm)
            args = dict(format=fmt)
            if fmt == 'png':
                args['dpi'] = 300
            args.update(kwargs)
            plt.savefig(path, **args)
            if self.verbose:
                logger.info(f'Saved figure to {s.i(rel_path(path))}')
