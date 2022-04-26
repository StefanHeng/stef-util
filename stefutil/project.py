"""
Project & project file structure related
"""

import os
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
    def __init__(
            self, base_path: str = None, project_dir: str = None, package_name: str = None,
            dataset_dir: str = None, model_dir: str = None
    ):
        """
        :param base_path: Absolute system path for root directory that contains a project folder & a data folder
        :param project_dir: Project root directory name that contains a folder for main source files
        :param package_name: python package/Module name which contain main source files
        :param dataset_dir: Directory name that contains datasets
        :param model_dir: Directory name that contains trained models
        """
        self.base_path = base_path
        self.proj_dir = project_dir
        self.pkg_nm = package_name
        self.dset_dir = dataset_dir
        self.model_dir = model_dir

        self.plot_path = os_join(self.base_path, self.proj_dir, 'plot')
        os.makedirs(self.plot_path, exist_ok=True)

    def save_fig(self, title, save=True):
        if save:
            fnm = f'{title}, {now(for_path=True)}.png'
            plt.savefig(os_join(self.plot_path, fnm), dpi=300)
