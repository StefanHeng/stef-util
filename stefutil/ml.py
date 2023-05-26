"""
machine learning
"""

import sys
import math
from typing import Tuple, Dict, Union, Iterable

import pandas as pd
import torch
from sklearn.metrics import classification_report

from stefutil.prettier import fmt_num

__all__ = ['model_param_size', 'get_model_num_trainable_parameter', 'is_on_colab', 'eval_array2report_df']


def get_torch_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def model_param_size(m: torch.nn.Module, as_str=True) -> Union[int, str]:
    num = m.num_parameters()
    assert num == sum(p.numel() for p in m.parameters())
    return fmt_num(num) if as_str else num


def get_model_num_trainable_parameter(model: torch.nn.Module, readable: bool = True) -> Union[int, str]:
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return fmt_num(n) if readable else n


def is_on_colab() -> bool:
    return 'google.colab' in sys.modules


def eval_array2report_df(
        labels: Iterable, preds: Iterable, report_args: Dict = None, pretty: bool = True
) -> Tuple[pd.DataFrame, float]:
    report = classification_report(labels, preds, **(report_args or dict()))
    if 'accuracy' in report:
        acc = report['accuracy']
    else:
        vals = [v for k, v in report['micro avg'].items() if k != 'support']
        assert all(math.isclose(v, vals[0], abs_tol=1e-8) for v in vals)
        acc = vals[0]
    return pd.DataFrame(report).transpose(), round(acc, 3) if pretty else acc
