from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ReduceLROnPlateau, TensorBoard, ModelCheckpoint
from tensorflow.keras.models import load_model
import argparse
import h5py
import os
import pprint
import numpy as np

from lib.models.srcnn import Srcnn
from lib.datasets import xray
from lib.config import cfg, cfg_from_file
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
    parser.add_argument('--start_epoch', dest='start_epoch',
                        help='starting epoch',
                        default=0, type=int)
    parser.add_argument('--epochs', dest='max_epochs',
                        help='number of epochs to train',
                        default=200, type=int)
    parser.add_argument('--disp_interval', dest='disp_interval',
                        help='number of iterations to display',
                        default=100, type=int)
    parser.add_argument('--save_dir', dest='save_dir',
                        help='directory to save models', default="models",
                        type=str)
    parser.add_argument('--nw', dest='num_workers',
                        help='number of worker to load data',
                        default=4, type=int)
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=64, type=int)
    parser.add_argument('--prebuilt', dest='prebuilt',
                        help='Whether dataset is converted to tf records already',
                        default=False, type=bool)
    parser.add_argument('--channels', dest='channels',
                        help='Number of channels',
                        default=1, type=int)
    parser.add_argument('--num_val', default=4000, type=int, 
                        help="Number of validation image")
    parser.add_argument('--model', default='srcnn', type=str, 
                        help="Model name")

    # config optimization
    parser.add_argument('--o', dest='optimizer',
                        help='training optimizer',
                        default="sgd", type=str)
    parser.add_argument('--lr', dest='lr',
                        help='starting learning rate',
                        default=0.0001, type=float)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                        help='learning rate decay ratio',
                        default=0.1, type=float)
    parser.add_argument('--session', dest='session',
                        help='Current session. Useful for multiple times training',
                        default=1, type=int)

    # resume trained model
    parser.add_argument('--r', dest='resume',
                        help='resume checkpoint or not',
                        default=False, type=bool)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint(weight file) to load model',
                        default="", type=str)

    args = parser.parse_args()
    return args

def rmse(y_true, y_pred):
    # Slightly different here. I just use the mean of batch
    # Not the sum over batch
    #return K.sqrt(K.mean(K.square(y_pred - y_true)))
    # Below should match the form on the project page
    return K.sum(K.sqrt(K.mean(K.square(y_pred - y_true), axis=[1,2,3])))

if __name__ == '__main__':

    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file:
        cfg_from_file(args.cfg_file)

    cfg.TRAIN.LEARNING_RATE = args.lr
    cfg.TRAIN.BATCH_SIZE = args.batch_size
    cfg.TRAIN.NUM_WORKERS = args.num_workers
    cfg.TRAIN.GAMMA = args.lr_decay_gamma

    cfg.DATA_DIR = os.path.join(cfg.DATA_DIR, args.dataset)
    assert os.path.isdir(cfg.DATA_DIR), "Couldn't find the dataset at {}".format(cfg.DATA_DIR)
    cfg.NUM_VAL = args.num_val
    cfg.CHANNELS = args.channels


    # Build dataset
    # If args.prebuilt == True, it will just return the number of samples in
    # 'train', 'val', and 'test'
    count_dict = build_dataset(args.prebuilt)
    print(count_dict)

    print('Using config:')
    pprint.pprint(cfg)

    np.random.seed(cfg.RNG_SEED)


    # Create model
    model = Srcnn(cfg.INPUT_IMAGE_SIZE, cfg.OUTPUT_LABEL_SIZE,
                  cfg.CHANNELS)
    # The compile step specifies the training configuration.
    if args.model == 'srcnn':
        optimizer = LRMultiplierSGD(lr=cfg.TRAIN.LEARNING_RATE,
                                    momentum=cfg.TRAIN.MOMENTUM,
                                    multipliers=[1, 1, 1, 1, 0.1, 0.1])
    else:
        optimizer = tf.train.MomentumOptimizer(cfg.TRAIN.LEARNING_RATE,
                                               cfg.TRAIN.MOMENTUM)
    if args.resume:
        args.start_epoch = int(args.checkpoint.split('/')[-1].split('-')[-2][-3:])
        model.load_weights(args.checkpoint)

    model.compile(optimizer=optimizer,
                  loss=tf.losses.mean_squared_error, 
                  metrics=[rmse])


    train_dataset = xray.get_dataset('train')
    val_dataset = xray.get_dataset('val')
    # Callback
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    save_filepath = os.path.join(args.save_dir,
                                 str(args.session)+"-weights-{epoch:03d}-{val_rmse:.2f}.ckpt")
    callbacks = [
        ReduceLROnPlateau(monitor='val_rmse', factor=cfg.TRAIN.GAMMA,
                          patience=3, mode='min', cooldown=1, 
                          min_delta=1e-3),
        TensorBoard(log_dir='./logs'),
        ModelCheckpoint(save_filepath, monitor='val_rmse', period=1,
                        save_best_only=True, mode='min', verbose=1,
                        save_weights_only=True)
    ]
    # Training
    model.fit(train_dataset, initial_epoch=args.start_epoch, epochs=args.max_epochs,
              steps_per_epoch=count_dict['train_count'] // cfg.TRAIN.BATCH_SIZE, 
              validation_data=val_dataset,
              validation_steps=count_dict['val_count'] // cfg.TRAIN.BATCH_SIZE,
              callbacks=callbacks)
