"""
For internal import check

Intended for skipping package imports on deep-learning libraries if they are not needed
        This will save the heavy loading times for packages like `torch`, `transformers`
"""


import os
import importlib.metadata
from typing import List

from stefutil.prettier import *


__all__ = ['_SU_USE_PLT', '_SU_USE_ML', '_SU_USE_DL']

_PKGS_PLT = ['matplotlib', 'seaborn']
_PKGS_ML = ['scikit-learn']
_PKGS_DL = ['torch', 'tensorboard', 'transformers', 'sentence-transformers', 'spacy']


_installed_pkgs = [(dist.metadata['Name'], dist.version) for dist in importlib.metadata.distributions()]
_installed_pkgs = set([name for (name, ver) in _installed_pkgs])


def check_use(flag_name: str = 'SU_USE_DL', desc: str = 'Deep Learning', expected_packages: List[str] = None) -> bool:
    # Whether to use certain utilities, based on the environment variable `SU_USE_<type>`
    flag = os.environ.get(flag_name, 'True')  # by default, import all packages
    ca.check_mismatch(display_name=f'`{flag_name}` Flag', val=flag, accepted_values=['True', 'False', 'T', 'F'])
    use = flag in ['True', 'T']

    if use:
        # check that the required packages for expected category of utility functions are in the environment
        pkgs_found = [pkg for pkg in expected_packages if pkg in _installed_pkgs]
        pkgs_missing = [pkg for pkg in expected_packages if pkg not in _installed_pkgs]
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


_SU_USE_PLT = check_use(flag_name='SU_USE_PLT', desc='Plotting', expected_packages=_PKGS_PLT)
_SU_USE_ML = check_use(flag_name='SU_USE_ML', desc='Machine Learning', expected_packages=_PKGS_ML)
_SU_USE_DL = check_use(flag_name='SU_USE_DL', desc='Deep Learning', expected_packages=_PKGS_DL)
