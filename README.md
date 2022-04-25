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
