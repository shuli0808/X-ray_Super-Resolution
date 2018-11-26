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

from lib.models.srcnn import Srcnn
from lib.datasets import xray
from lib.config import cfg, get_output_dir
from lib.utils import build_dataset


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='xray_images', type=str)
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

if __name__ == '__main__':

    args = parse_args()

    print('Called with args:')
    print(args)


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

    print('Using config:')
    pprint.pprint(cfg)
    np.random.seed(cfg.RNG_SEED)


    model = load_model(args.checkpoint)


    test_dataset = xray.get_dataset('test')
    # Testing
    result = model.predict(test_dataset)

    #print(result.shape)
    assert result.shape == (count_dict['test_count'], cfg.OUTPUT_LABEL_SIZE,
                            cfg.OUTPUT_LABEL_SIZE, cfg.CHANNELS), \
            'result has shape {} != {}'.format(result.shape, (count_dict['test_count'],
                           cfg.OUTPUT_LABEL_SIZE, cfg.OUTPUT_LABEL_SIZE,
                           cfg.CHANNELS))

    output_dir = get_output_dir(args.model)
    test_images_dir = os.path.join(cfg.DATA_DIR, 'test_images_64x64')
    test_image_filenames = np.array([os.path.join(test_images_dir, f) for f in 
                                     filenames if f.endswith('.png')])
    for i in range(result.shape[0]):
        filename_suffix = test_image_filenames[i].split('/')[-1]
        # This will rescale image values to be within [0, 255]
        save_img(os.path.join(output_dir, filename_suffix), result[i], 
                 data_format='channels_last', scale=True)


