import os
import os.path as osp
import numpy as np
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

__C = edict()

cfg = __C

# Training options
__C.TRAIN = edict()
# Initial learning rate
__C.TRAIN.LEARNING_RATE = 0.001

# Momentum
__C.TRAIN.MOMENTUM = 0.9

# Weight decay, for regularization
__C.TRAIN.WEIGHT_DECAY = 0.0005

# Factor for reducing the learning rate
__C.TRAIN.GAMMA = 0.1

# Minibatch size 
__C.TRAIN.BATCH_SIZE = 128

# Use augmentation
__C.TRAIN.AUGMENT = False

# Parallel calls
__C.TRAIN.NUM_WORKERS = 4


# Number of validation images
#__C.NUM_VAL = 5000 
#__C.NUM_TRAIN = 10000 
#__C.NUM_TEST = 5000 
__C.TEST = edict()
# Minibatch size 
__C.TEST.BATCH_SIZE = 129


# For reproducibility
__C.RNG_SEED = 3
# Root directory of project
__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..'))
# Data directory
__C.DATA_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'data'))
# Place outputs under an experiments directory
__C.EXP_DIR = 'default'

# GrayImage or RGB
__C.CHANNELS = 1
__C.INPUT_IMAGE_SIZE = 32
__C.OUTPUT_LABEL_SIZE = 20
#__C.OUTPUT_START_INDEX = 6

# Crop and resize
#number of pixel of each block after cropping
__C.ORIGIN_OUTPUT_LABEL_SIZE = 128


def get_output_tb_dir(dataset_name, weights_filename):
    """Return the directory where tensorflow summaries are placed.
    If the directory does not exist, it is created.
    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    outdir = osp.abspath(osp.join(__C.ROOT_DIR, 'tensorboard', __C.EXP_DIR,
                                  dataset_name))
    if weights_filename is None:
        weights_filename = 'default'
    outdir = osp.join(outdir, weights_filename)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir

def get_output_dir(dataset_name, weights_filename=None):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.
    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    outdir = osp.abspath(osp.join(__C.ROOT_DIR, 'output', __C.EXP_DIR,
                                  dataset_name))
    if weights_filename is None:
        weights_filename = 'default'
    outdir = osp.join(outdir, weights_filename)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print(('Error under config key: {}'.format(k)))
                raise
        else:
            b[k] = v

def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)


def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert subkey in d
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), \
                'type {} does not match original type {}'.format(
                    type(value), type(d[subkey]))
        d[subkey] = value

