"""
Deep-Learning related prettier & prettier logging
"""

import logging
from typing import List, Dict, Iterable, Any, Union
from collections.abc import Sized

from tqdm.auto import tqdm

from stefutil.prettier import s, ca
from stefutil.packaging import _use_dl


__all__ = ['MlPrettier']


if _use_dl():
    from transformers import Trainer, TrainerCallback

    __all__ += ['MyProgressCallback', 'LogStep']


class MlPrettier:
    """
    My utilities for deep learning training logging
    """
    no_prefix = ('epoch', 'global_step', 'step')  # order matters, see `single`

    def __init__(
            self, ref: Dict[str, Any] = None, metric_keys: List[str] = None, no_prefix: Iterable[str] = no_prefix,
            with_color: bool = False
    ):
        """
        :param ref: Reference that are potentially needed
            i.e. for logging epoch/step, need the total #
        :param metric_keys: keys that are considered metric
            Will be logged in [0, 100]
        """
        self.ref = ref
        self.metric_keys = metric_keys or ['acc', 'precision', 'recall', 'f1', 'auc']
        self.no_prefix = no_prefix
        self.with_color = with_color

    def __call__(self, d: Union[str, Dict], val=None) -> Union[Any, Dict[str, Any]]:
        """
        :param d: If str, prettify a single value
            Otherwise, prettify a dict
        """
        is_dict = isinstance(d, dict)
        if not ((isinstance(d, str) and val is not None) or is_dict):
            raise ValueError('Either a key-value pair or a mapping is expected')
        if is_dict:
            d: Dict
            return {k: self.single(k, v) for k, v in d.items()}
        else:
            return self.single(d, val)

    def single(self, key: str = None, val: Any = None) -> Union[str, List[str], Dict[str, Any]]:
        """
        `val` processing is infered based on key
        """
        if key in MlPrettier.no_prefix:
            k = next(iter(k for k in self.ref.keys() if key in k))
            lim = self.ref[k]
            assert isinstance(val, (int, float))
            len_lim = len(str(lim))
            if isinstance(val, int):
                s_val = f'{val:>{len_lim}}'
            else:
                fmt = f'%{len_lim + 4}.3f'
                s_val = fmt % val
            if self.with_color:
                return f'{s.i(s_val)}/{s.i(lim)}'
            else:
                return f'{s_val}/{lim}'  # Pad integer
        elif 'loss' in key:
            return f'{round(val, 4):7.4f}'
        elif any(k in key for k in self.metric_keys):  # custom in-key-ratio metric
            def _single(v):
                return f'{round(v * 100, 2):6.2f}' if v is not None else '-'

            if isinstance(val, list):
                return [_single(v) for v in val]
            elif isinstance(val, dict):
                return {k: _single(v) for k, v in val.items()}
            else:
                return _single(val)
        elif 'learning_rate' in key or 'lr' in key:
            return f'{round(val, 7):.3e}'
        elif 'perplexity' or 'ppl' in key:
            return f'{round(val, 2):.2f}'
        else:
            return val

    def should_add_split_prefix(self, key: str) -> bool:
        """
        Whether to add split prefix to the key
        """
        return key not in self.no_prefix

    def add_split_prefix(self, d: Dict[str, Any], split: str = None):
        if split is None:
            return d
        else:
            return {f'{split}/{k}' if self.should_add_split_prefix(k) else k: v for k, v in d.items()}


if _use_dl():
    class MyProgressCallback(TrainerCallback):
        """
        My modification to the HF progress callback

        1. Effectively remove all logging, keep only the progress bar w.r.t. this callback
        2. Train tqdm for each epoch only
        3. Option to disable progress bar for evaluation

        Expects to start from whole epochs
        """
        def __init__(self, train_only: bool = False):
            """
            :param train_only: If true, disable progress bar for evaluation
            """
            self.training_bar = None
            self.prediction_bar = None

            self.train_only = train_only
            self.step_per_epoch = None
            self.current_step = None

        @staticmethod
        def _get_steps_per_epoch(state):
            assert state.max_steps % state.num_train_epochs == 0
            return state.max_steps // state.num_train_epochs

        @staticmethod
        def _get_curr_epoch(state, is_eval: bool = False) -> str:
            n_ep = int(state.epoch)
            if not is_eval:  # heuristic judging by the eval #epoch shown
                n_ep += 1

            return MlPrettier(ref=dict(epoch=state.num_train_epochs), with_color=True)('epoch', n_ep)

        def on_epoch_begin(self, args, state, control, **kwargs):
            if state.is_local_process_zero:
                if not self.step_per_epoch:
                    self.step_per_epoch = MyProgressCallback._get_steps_per_epoch(state)
                ep = MyProgressCallback._get_curr_epoch(state)
                self.training_bar = tqdm(total=self.step_per_epoch, desc=f'Train Epoch {ep}', unit='ba')
            self.current_step = 0

        def on_train_begin(self, args, state, control, **kwargs):
            pass

        def on_epoch_end(self, args, state, control, **kwargs):
            if state.is_local_process_zero:
                self.training_bar.close()
                self.training_bar = None

        def on_step_end(self, args, state, control, **kwargs):
            if state.is_local_process_zero:
                self.training_bar.update(1)

        def on_prediction_step(self, args, state, control, eval_dataloader=None, **kwargs):
            if not self.train_only:
                if state.is_local_process_zero and isinstance(eval_dataloader.dataset, Sized):
                    if self.prediction_bar is None:
                        ep = MyProgressCallback._get_curr_epoch(state, is_eval=True)
                        desc = f'Eval Epoch {ep}'
                        self.prediction_bar = tqdm(
                            desc=desc, total=len(eval_dataloader), leave=self.training_bar is None, unit='ba'
                        )
                    self.prediction_bar.update(1)

        def on_evaluate(self, args, state, control, **kwargs):
            if not self.train_only:
                if state.is_local_process_zero:
                    if self.prediction_bar is not None:
                        self.prediction_bar.close()
                    self.prediction_bar = None

        def on_log(self, args, state, control, logs=None, **kwargs):
            if state.is_local_process_zero and self.training_bar is not None:
                _ = logs.pop("total_flos", None)

        def on_train_end(self, args, state, control, **kwargs):
            pass

        @staticmethod
        def get_current_progress_bar(trainer: Trainer):
            """
            Intended for adding per-step metrics to the progress bar during HF training

            This is a hack,
                since HF API don't support per-step callback not to mention exposing those metrics to the progress bar
            """
            callback = next(cb for cb in trainer.callback_handler.callbacks if isinstance(cb, MyProgressCallback))
            return callback.training_bar if trainer.model.training else callback.prediction_bar


    class LogStep:
        """
        My typical terminal, file & tqdm logging for a single step
        """
        def __init__(
                self, trainer: Trainer = None, pbar: tqdm = None, prettier: MlPrettier = None,
                logger: logging.Logger = None, file_logger: Union[logging.Logger, bool] = None,
                tb_writer=None, trainer_with_tqdm: bool = True,
                global_step_with_epoch: bool = True, prettier_console: bool = False, console_with_split: bool = False
        ):
            self.trainer = trainer
            self.trainer_with_tqdm = False
            if trainer is not None:
                if hasattr(trainer, 'with_tqdm'):
                    self.trainer_with_tqdm = trainer.with_tqdm
                else:
                    self.trainer_with_tqdm = trainer_with_tqdm

            self.pbar = None
            if trainer:
                assert not pbar  # sanity check
            else:
                self.pbar = pbar

            self.prettier = prettier or MlPrettier()
            self.logger = logger
            self.file_logger, self.logger_logs_file = None, False
            if file_logger is True:  # assumes `logger` also writes to file
                self.logger_logs_file = True
            elif isinstance(file_logger, logging.Logger):
                self.file_logger = file_logger
            self.tb_writer = tb_writer  # a `torch.utils.tensorboard.SummaryWriter` or a `tensorboardX.SummaryWriter`

            self.global_step_with_epoch = global_step_with_epoch
            self.prettier_console = prettier_console
            self.console_with_split = console_with_split

        def _should_add(self, key: str) -> bool:
            return self.prettier.should_add_split_prefix(key) if self.prettier else True

        def __call__(
                self, d_log: Dict, training: bool = None, to_console: bool = True, split: str = None, prefix: str = None,
                add_pbar_postfix: bool = True, to_file: bool = True
        ):
            """
            :param d_log: Dict to log
            :param training: Whether `d_log` is for training or evaluation
            :param to_console: Whether to log to console
            :param split: If specified, one of [`train`, `eval`, `test`]
                Overrides `training`
            :param prefix: If specified, prefix is inserted before the log
            """
            if split is None:
                if training is not None:
                    training = training
                else:
                    training = self.trainer.model.training
                split_str = 'train' if training else 'eval'
            else:
                ca.assert_options('Train Mode', split, ['train', 'eval', 'dev', 'test'])
                training = split == 'train'
                split_str = split
            d_log_p = self.prettier(d_log) if self.prettier else d_log

            if self.tb_writer:
                if self.global_step_with_epoch:
                    tb_step = d_log['step'] if training else d_log['epoch']
                else:
                    tb_step = d_log.get('global_step', None) or d_log['step']  # at least one of them must exist
                for k, v in d_log.items():
                    if self._should_add(k):
                        self.tb_writer.add_scalar(tag=f'{split_str}/{k}', scalar_value=v, global_step=tb_step)

            if (self.trainer is not None and self.trainer_with_tqdm) or self.pbar is not None:  # a custom field I added
                if self.pbar is not None:
                    pbar = self.pbar
                else:
                    pbar = MyProgressCallback.get_current_progress_bar(self.trainer)
                if pbar and add_pbar_postfix:
                    tqdm_kws = {k: s.i(v) for k, v in d_log_p.items() if self._should_add(k)}
                    pbar.set_postfix(tqdm_kws)
            if to_console and self.logger:
                d = d_log_p if self.prettier_console else d_log
                if self.console_with_split and split_str:
                    d = self.prettier.add_split_prefix(d, split=split_str)
                msg = s.i(d)
                if prefix:
                    msg = f'{prefix}{msg}'

                extra = None
                if self.logger_logs_file and not to_file:  # blocks logging to file
                    extra = dict(block='file')
                self.logger.info(msg, extra=extra)

            if to_file:
                msg = s.nc(d_log)
                if prefix:
                    msg = f'{prefix}{msg}'

                if self.file_logger:
                    self.file_logger.info(msg)
                elif self.logger_logs_file and self.logger and not to_console:
                    # if `to_console` is true, already logged to file too
                    extra = dict(block='stdout')  # blocks logging to console
                    self.logger.info(msg, extra=extra)


if __name__ == '__main__':
    from stefutil.prettier import sic

    def check_prettier():
        mp = MlPrettier(ref=dict(epoch=3, step=3, global_step=9))
        sic(mp.single(key='global_step', val=4))
        sic(mp.single(key='step', val=2))
    check_prettier()
