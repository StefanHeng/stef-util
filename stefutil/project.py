"""
Project & project file structure related
"""

import json
from os.path import join as os_join

import matplotlib.pyplot as plt

from stefutil.container import get
from stefutil.prettier import now


__all__ = ['StefConfig', 'StefUtil']


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
