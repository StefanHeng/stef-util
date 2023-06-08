"""
machine learning
"""

import sys
import math
from typing import Tuple, Dict, Union, Iterable

import pandas as pd
import torch
from sklearn.metrics import classification_report

from stefutil.prettier import fmt_num, fmt_sizeof, ca

__all__ = [
    'get_torch_device',
    'model_param_size', 'get_model_num_trainable_parameter',
    'get_trainable_param_meta', 'get_model_size', 'get_model_meta',
    'is_on_colab', 'eval_array2report_df'
]


def get_torch_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def model_param_size(m: torch.nn.Module, as_str=True) -> Union[int, str]:
    num = m.num_parameters()
    assert num == sum(p.numel() for p in m.parameters())
    return fmt_num(num) if as_str else num


def get_model_num_trainable_parameter(model: torch.nn.Module, readable: bool = True) -> Union[int, str]:
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return fmt_num(n) if readable else n


def get_trainable_param_meta(model: torch.nn.Module, fmt='str'):
    """
    Edited from `PeftModel.get_trainable_parameters`
    """
    ca.check_mismatch('#Param Format', fmt, ['int', 'str'])

    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    ratio = round(trainable_params / all_param * 100, 2)
    if fmt == 'str':
        trainable_params = fmt_num(trainable_params)
        all_param = fmt_num(all_param)
    return {
        '#trainable': trainable_params,
        '#all': all_param,
        'ratio': f'{ratio}%'
    }


def get_model_size(model: torch.nn.Module, fmt='str', all_only: bool = True):
    ca.check_mismatch('Size Format', fmt, ['int', 'str'])

    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    ret = dict(param_size=param_size, buffer_size=buffer_size, size_all=param_size + buffer_size)
    if fmt == 'str':
        ret = {k: fmt_sizeof(v) for k, v in ret.items()}
    return ret['size_all'] if all_only else ret


def get_model_meta(model: torch.nn.Module):
    return dict(param=get_trainable_param_meta(model), size=get_model_size(model))


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
