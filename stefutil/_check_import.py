"""
For internal import check

Intended for skipping package imports on deep-learning libraries if they are not needed
        This will save the heavy loading times for packages like `torch`, `transformers`
"""


import os
import importlib.metadata
from typing import List

from stefutil.prettier import *


__all__ = ['_use_plot', '_use_ml', '_use_dl']

_PKGS_PLT = ['matplotlib', 'seaborn']
_PKGS_ML = ['scikit-learn']
_PKGS_DL = ['torch', 'tensorboard', 'transformers', 'sentence-transformers', 'spacy']


def check_use(flag_name: str = 'SU_USE_DL', desc: str = 'Deep Learning', expected_packages: List[str] = None) -> bool:
    # Whether to use certain utilities, based on the environment variable `SU_USE_<type>`
    flag = os.environ.get(flag_name, 'True')  # by default, import all packages
    ca.assert_options(display_name=f'`{flag_name}` Flag', val=flag, options=['True', 'False', 'T', 'F'])
    use = flag in ['True', 'T']

    if use:
        # lazy check to save time
        if not hasattr(check_use, '_INSTALLED_PACKAGES'):
            pkgs = [(dist.metadata['Name'], dist.version) for dist in importlib.metadata.distributions()]
            check_use.INSTALLED_PACKAGES = set([name for (name, ver) in pkgs])

        # check that the required packages for expected category of utility functions are in the environment
        pkgs_found = [pkg for pkg in expected_packages if pkg in check_use.INSTALLED_PACKAGES]
        pkgs_missing = [pkg for pkg in expected_packages if pkg not in check_use.INSTALLED_PACKAGES]
        if len(pkgs_missing) > 0:

            if len(expected_packages) > 1:
                msg = f'packages are'
                d_log = {'dl-packages-expected': expected_packages, 'dl-packages-found': pkgs_found, 'dl-packages-missing': pkgs_missing}
                pkg = f'Please install the following packages: {pl.i(d_log)}'
            else:
                msg = f'package is'
                pkg = f'Please install the package {pl.i(expected_packages[0])}.'
            raise ImportError(f'{desc} {msg} not found in the environment when `{flag_name}` is set to True. {pkg}')
    return use


def _use_plot():
    return check_use(flag_name='SU_USE_PLT', desc='Plotting', expected_packages=_PKGS_PLT)


def _use_ml():
    return check_use(flag_name='SU_USE_ML', desc='Machine Learning', expected_packages=_PKGS_ML)


def _use_dl():
    return check_use(flag_name='SU_USE_DL', desc='Deep Learning', expected_packages=_PKGS_DL)
