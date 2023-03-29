# Stef-util
My utility functions to start machine learning projects 

## Usage 
```python
from stefutil import *
# Change those to your project
from os.path import join as os_join
BASE_PATH = '/Users/stefanh/Documents/UMich/Research/Clarity Lab/Zeroshot Text Classification'
PROJ_DIR = 'Zeroshot-Text-Classification'
PKG_NM = 'zeroshot_encoder'
DSET_DIR = 'dataset'
MODEL_DIR = 'models'

# Setup project-level functions
# Have a config `json` file ready
sconfig = StefConfig(config_file=os_join(BASE_PATH, PROJ_DIR, PKG_NM, 'util', 'config.json')).__call__
_sutil = StefUtil(
    base_path=BASE_PATH, project_dir=PROJ_DIR, package_name=PKG_NM, dataset_dir=DSET_DIR, model_dir=MODEL_DIR
)
save_fig = _sutil.save_fig
# Now you can call `sconfig` and `save_fig`

# Set argument "enum" checks
ca.cache_mismatch(
    'Bar Plot Orientation', attr_name='bar_orient', accepted_values=['v', 'h', 'vertical', 'horizontal']
)
# Now you can call `ca` like so:
ori = 'v'
ca(bar_orient=ori)
```



## Highlights

An incomplete list of features. 



### Modified `IceCream` Debugging

[`IceCream`](https://github.com/gruns/icecream) debugging print with custom output width, intended for various terminal sizes. 

```python
from stefutil import mic
mic.output_width = 256 
```



### Custom colored logger formatting 

```python
from stefutil import get_logger
logger = get_logger('<Script-Name>')
```



### Prettier Training Log

Updates to [`HuggingFace transformers`](https://github.com/huggingface/transformers) training & logging, including custom [`tqdm`](https://github.com/tqdm/tqdm) progress bar and [`TensorBoard`](https://www.tensorflow.org/tensorboard) logging. 

```python
from stefutil import MlPrettier, LogStep, pl

# Before training
prettier = MlPrettier(ref=train_meta, metric_keys=['acc', 'recall', 'auc', 'ikr'])
ls = LogStep(
    trainer=trainer, prettier=prettier, logger=logger, 
    file_logger=file_logger, tb_writer=tb_writer
)

"""
...
"""

# During each training/eval step
for step in training_or_eval_steps:
    logs = prettier(logs)
    logger.info(pl.i(logs))
    file_logger.info(pl.nc(logs))
```



