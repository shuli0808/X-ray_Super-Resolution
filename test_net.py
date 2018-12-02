from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import array_to_img as array_to_img
from tensorflow.keras.preprocessing.image import save_img as save_img
import argparse
import h5py
from PIL import Image
import os
import pprint
import numpy as np
from tqdm import tqdm

from lib.models.srcnn import Srcnn
from lib.datasets import xray
from lib.config import cfg, get_output_dir
from lib.utils import build_dataset
from lib.models.LRMultiplierSGD import LRMultiplierSGD


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='xray_images', type=str)
    parser.add_argument('--cfg_file', dest='cfg_file',
                        help='path to cfg_file (.yaml)',
                        default='', type=str)
    parser.add_argument('--nw', dest='num_workers',
                        help='number of worker to load data',
                        default=4, type=int)
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=128, type=int)
    parser.add_argument('--prebuilt', dest='prebuilt',
                        help='Whether prebuilt dataset to tf record',
                        default=True, type=bool)
    parser.add_argument('--channels', dest='channels',
                        help='Number of channels',
                        default=3, type=int)
    parser.add_argument('--num_val', default=5000, type=int, 
                        help="Number of validation image")
    parser.add_argument('--model', default='srcnn', type=str, 
                        help="Model name")

    # resume trained model
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint(weight file) to load model',
                        default="", type=str)

    args = parser.parse_args()
    return args

def rmse(y_true, y_pred):
    # Slightly different here. I just use the mean of batch
    # Not the sum over batch
    return K.sqrt(K.mean(K.square(y_pred - y_true)))
    # Below should match the form on the project page
    #return K.sum(K.sqrt(K.mean(K.square(y_pred - y_true), axis=[1,2,3])))

if __name__ == '__main__':

    args = parse_args()

    print('Called with args:')
    print(args)
    if args.cfg_file:
        cfg_from_file(args.cfg_file)


    cfg.TRAIN.BATCH_SIZE = args.batch_size
    cfg.TRAIN.NUM_WORKERS = args.num_workers

    cfg.DATA_DIR = os.path.join(cfg.DATA_DIR, args.dataset)
    assert os.path.isdir(cfg.DATA_DIR), "Couldn't find the dataset at {}".format(cfg.DATA_DIR)
    cfg.NUM_VAL = args.num_val
    cfg.CHANNELS = args.channels

    # Build dataset
    # If args.prebuilt == True, it will just return the number of samples in
    # 'train', 'val', and 'test'
    count_dict = build_dataset(args.prebuilt)

    print('Using config:')
    pprint.pprint(cfg)

    np.random.seed(cfg.RNG_SEED)


    #model = load_model(args.checkpoint)
    # Create model
    model = Srcnn(cfg.INPUT_IMAGE_SIZE, cfg.OUTPUT_LABEL_SIZE,
                  cfg.CHANNELS, False)
    # The compile step specifies the training configuration.
    if args.model == 'srcnn':
        optimizer = LRMultiplierSGD(lr=cfg.TRAIN.LEARNING_RATE,
                                    momentum=cfg.TRAIN.MOMENTUM,
                                    multipliers=[1, 1, 1, 1, 0.1, 0.1])
    else:
        optimizer = tf.train.MomentumOptimizer(cfg.TRAIN.LEARNING_RATE,
                                               cfg.TRAIN.MOMENTUM)

    model.load_weights(args.checkpoint)
    model.compile(optimizer=optimizer,
                  loss=tf.losses.mean_squared_error, 
                  metrics=[rmse])


    test_dataset = xray.get_dataset('test')
    # Testing
    result = model.predict(test_dataset, 
                           steps=int(count_dict['test_count'] /
                                     cfg.TEST.BATCH_SIZE),
                           verbose=1)

    #print(result.shape)
    assert result.shape == (count_dict['test_count'],
                            cfg.ORIGIN_OUTPUT_LABEL_SIZE,
                            cfg.ORIGIN_OUTPUT_LABEL_SIZE, cfg.CHANNELS), \
            'result has shape {} != {}'.format(result.shape, (count_dict['test_count'],
                           cfg.ORIGIN_OUTPUT_LABEL_SIZE, cfg.ORIGIN_OUTPUT_LABEL_SIZE,
                           cfg.CHANNELS))

    output_dir = get_output_dir(args.model)
    test_images_dir = os.path.join(cfg.DATA_DIR, 'test_images_64x64')
    test_image_filenames = np.array([os.path.join(test_images_dir, f) for f in 
                                     filenames if f.endswith('.png')])
    for i in tqdm(range(result.shape[0])):
        filename_suffix = test_image_filenames[i].split('/')[-1]
        # This will rescale image values to be within [0, 255]
        save_img(os.path.join(output_dir, filename_suffix), result[i], 
                 data_format='channels_last', scale=True)


