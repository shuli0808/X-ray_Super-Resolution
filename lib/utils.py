"""Split the Xray images dataset into train/dev/test. 
The Xray dataset comes in the following format:
    train_images_64x64/
        train_xxxxx.png
        ...
    train_images_128x128/
        train_xxxxx.png
        ...
    test_images64x64/
        test_xxxxx.png
        ...
We already have a test set created, so we only need to split
"train_images_64x64" and the corresponding labels in
"train_images_128x128" into train and val sets.
We'll take args.num_val (5000) of "train_images_xxx" as val set.
"""

import os
import sys
import numpy as np

from PIL import Image
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img as load_img
from tensorflow.keras.preprocessing.image import img_to_array as img_to_array
from lib.config import cfg, cfg_from_file, cfg_from_list

# Serialize images, together with labels, to TF records
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))



def build_dataset(prebuilt=False):
    if prebuilt:
        # Just return the number of training/validation/testing instances
        # This need to be check later whether it is correct
        count_dict = {}
        for split in ['train', 'val', 'test']:
            tf_records_filename = os.path.join(cfg.DATA_DIR, split+'.tfrecord')

            count = 0
            for record in tf.python_io.tf_record_iterator(tf_records_filename):
                count += 1
            count_dict[split+'_count'] = count
        return count_dict

    # Define the data directories
    images_dir = os.path.join(cfg.DATA_DIR, 'train_images_64x64')
    labels_dir = os.path.join(cfg.DATA_DIR, 'train_images_128x128')
    test_images_dir = os.path.join(cfg.DATA_DIR, 'test_images_64x64')

    # Get the filenames in each directory (train and test)
    filenames = os.listdir(images_dir)
    image_filenames = np.array([os.path.join(images_dir, f) for f in
                                filenames if f.endswith('.png')])

    filenames = os.listdir(test_images_dir)
    test_image_filenames = np.array([os.path.join(test_images_dir, f) for f in 
                                     filenames if f.endswith('.png')])

    print('Original dataset contains {:d} training images, and {:d} testing imaegs'.format(
        len(image_filenames), len(test_image_filenames)))

    # Split the images in 'train_images'
    # Make sure to always shuffle with a fixed seed so that the split is reproducible
    #np.random.seed(cfg.RNG_SEED)
    rand_ind = np.random.permutation(len(image_filenames))
    val_ind = rand_ind[:cfg.NUM_VAL]
    train_ind = rand_ind[cfg.NUM_VAL:]

    val_image_filenames = image_filenames[val_ind]
    train_image_filenames = image_filenames[train_ind]

    print('We split dataset into {:d} training images, '
          '{:d} validation images and {:d} testing imaegs'.format(len(train_image_filenames),
                                          len(val_image_filenames), 
                                          len(test_image_filenames)))


    filenames = {'train_images': train_image_filenames,
                 'val_images': val_image_filenames,
                 'test_images': test_image_filenames}


    # Preprocess train, val and test
    for split in ['train', 'val', 'test']:
        tf_records_filename = os.path.join(cfg.DATA_DIR, split+'.tfrecord')
        writer = tf.python_io.TFRecordWriter(tf_records_filename)

        print("Processing {} images, saving preprocessed images to {}".format(split, tf_records_filename))
        color_mode = 'grayscale' if cfg.CHANNELS==1 else 'rgb'
        for filename in tqdm(filenames[split+'_images']):
            # Can optionally resize
            # Also make the range from [0, 255] -> [0, 1.0]
            img = img_to_array(load_img(filename, color_mode=color_mode, 
                                        target_size=(cfg.ORIGIN_OUTPUT_LABEL_SIZE,
                                                    cfg.ORIGIN_OUTPUT_LABEL_SIZE),
                                        interpolation= 'bicubic')) / 255.0
            if split != 'test':
                # load label
                filename_suffix = filename.split('/')[-1]
                label_path = os.path.join(labels_dir, filename_suffix)
                label = img_to_array(load_img(label_path, color_mode=color_mode,
                                              target_size=None)) / 255.0
                # Maybe implement crop to patches here
                crop_per_im = int((cfg.ORIGIN_OUTPUT_LABEL_SIZE / cfg.INPUT_IMAGE_SIZE) ** 2)
                sub_img = img.reshape(crop_per_im, cfg.INPUT_IMAGE_SIZE, cfg.INPUT_IMAGE_SIZE, cfg.CHANNELS)
                #Array that containes the block pixels
                sub_label = label.reshape(crop_per_im, cfg.INPUT_IMAGE_SIZE, cfg.INPUT_IMAGE_SIZE, cfg.CHANNELS)
                output_end_index = cfg.OUTPUT_START_INDEX + cfg.OUTPUT_LABEL_SIZE
                sub_label = sub_label[:, cfg.OUTPUT_START_INDEX:output_end_index,
                                      cfg.OUTPUT_START_INDEX:output_end_index, :]

                for b in range(crop_per_im):
                    # crop_and_save(imaeg, output_dir_split, label, size=64)
                    # Below two lines should also be in the crop_and_save function
                    example = tf.train.Example(features=tf.train.Features(
                        feature={'image': _bytes_feature(sub_img[b].tostring()),
                                 'label': _bytes_feature(sub_label[b].tostring())}
                    ))

                    writer.write(example.SerializeToString())
            else:
                example = tf.train.Example(features=tf.train.Features(
                    feature={'image': _bytes_feature(img.tostring())}
                ))

                writer.write(example.SerializeToString())
        writer.close()


    print("Done building dataset")
    return {'train_count': len(train_image_filenames) * crop_per_im, 
            'val_count': len(val_image_filenames) * crop_per_im, 
            'test_count': len(test_image_filenames)}


