"""Create the input data pipeline using `tf.data`"""

import tensorflow as tf
import os
from lib.config import cfg

featdef = {
    'image': tf.FixedLenFeature(shape=[], dtype=tf.string),
    'label': tf.FixedLenFeature(shape=[], dtype=tf.string),
}

def _parse_record(example_proto, is_test=False):
    """Parse a single record into image, and labels (channel_last)"""
    example = tf.parse_single_example(example_proto, featdef)
    if not is_test:
        im = tf.decode_raw(example['image'], tf.float32)
        im = tf.reshape(im, (cfg.INPUT_IMAGE_SIZE, cfg.INPUT_IMAGE_SIZE,
                             cfg.CHANNELS))
        label = tf.decode_raw(example['label'], tf.float32)
        label = tf.reshape(label, (cfg.INPUT_IMAGE_SIZE, cfg.INPUT_IMAGE_SIZE,
                                   cfg.CHANNELS))
        return im, label
    else:
        im = tf.decode_raw(example['image'], tf.float32)
        im = tf.reshape(im, (cfg.ORIGIN_OUTPUT_LABEL_SIZE,
                             cfg.ORIGIN_OUTPUT_LABEL_SIZE,
                             cfg.CHANNELS))
        # Need this to work for predict
        dummy = tf.decode_raw(example['label'], tf.float32)
        dummy = tf.reshape(dummy, (cfg.ORIGIN_OUTPUT_LABEL_SIZE,
                                   cfg.ORIGIN_OUTPUT_LABEL_SIZE,
                                   cfg.CHANNELS))
        return im, dummy



def get_dataset(mode):
    """Get the xray dataset.
    The filenames have format "{split}_{xxxxx}.png".
    For instance: "data_dir/train_04000.png".
    Args:
        mode: (str) whether to use the train, val, or test pipeline.
                     At training, we shuffle the data and have multiple epochs
    """
    # Create a Dataset serving batches of images and labels
    if mode == 'train':
        dataset = (tf.data.TFRecordDataset(os.path.join(cfg.DATA_DIR,
                                                        mode+'.tfrecord'))
            .map(_parse_record, num_parallel_calls=cfg.TRAIN.NUM_WORKERS)
            # Not sure whether we should use data augmentation
            #.map(train_fn, num_parallel_calls=cfg.TRAIN.NUM_WORKERS)
            .shuffle(12000)  # can change this number
            .batch(cfg.TRAIN.BATCH_SIZE)
            .repeat()
            .prefetch(1)  # make sure you always have one batch ready to serve
        )
    elif mode == 'val':
        dataset = (tf.data.TFRecordDataset(os.path.join(cfg.DATA_DIR,
                                                        mode+'.tfrecord'))
            .map(_parse_record)
            .batch(cfg.TRAIN.BATCH_SIZE)
            .prefetch(1)  # make sure you always have one batch ready to serve
        )
    elif mode == 'test':
        # Somewhat different in map
        dataset = (tf.data.TFRecordDataset(os.path.join(cfg.DATA_DIR,
                                                        mode+'.tfrecord'))
            .map(lambda e: _parse_record(e, True))
            .batch(cfg.TEST.BATCH_SIZE)
            .prefetch(1)  # make sure you always have one batch ready to serve
        )
    else:
        raise ValueError('mode {:s} not supported'.format(mode))

    # Create reinitializable iterator from dataset
    #iterator = dataset.make_initializable_iterator()
    # Note that for test, it would return None
    #images, labels = iterator.get_next()
    #iterator_init_op = iterator.initializer

    #inputs = {'images': images, 'labels': labels, 'iterator_init_op': iterator_init_op}
    #return inputs
    return dataset
