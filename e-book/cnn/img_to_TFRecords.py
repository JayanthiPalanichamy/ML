# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import ijson
from PIL import Image
import os 
import skimage.io as io

tf.reset_default_graph()

IMAGE_HEIGHT = 120
IMAGE_WIDTH = 120

FILE_PATH = '/Users/vijay/workspace/research/machineLearning/datasets/street2shop/meta/json/'
DATA_PATH = '/Users/vijay/workspace/research/machineLearning/datasets/street2shop/wtbi_tops_query_crop/'

tfrecords_filename = 'test_tops.tfrecords'


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _image_files_to_TFRecords(file_name, tfrecords_name):
    #column_names = ['photo', 'product']
    if os.path.isfile(tfrecords_name):
        print('The file already there, not doing anything!')
        return
    writer = tf.python_io.TFRecordWriter(tfrecords_name)

    with open(file_name, 'r') as f:
        objects = ijson.items(f, '')
        columns = list(objects)[0]
        for row in columns:
            img_path = DATA_PATH + np.str(row['photo']) + '.jpg'
            img = np.array(Image.open(img_path))
            height = img.shape[0]
            width = img.shape[1]
            img_raw = img.tostring()
            prod_id = row['product']
            example = tf.train.Example(features=tf.train.Features(feature={
                    'height': _int64_feature(height),
                    'width': _int64_feature(width),
                    'image_raw': _bytes_feature(img_raw),
                    'label': _int64_feature(prod_id)}))
            writer.write(example.SerializeToString())
        writer.close()
        
_image_files_to_TFRecords(FILE_PATH + 'test_pairs_tops.json',
                                              tfrecords_filename)

def read_and_decode(filename_queue, batch_size):
    
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'image_raw': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64)
        })

    # Convert from a scalar string tensor (whose single string has
    # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
    # [mnist.IMAGE_PIXELS].
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    label = tf.cast(features['label'], tf.int32)    
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    
    image = tf.reshape(image, [height, width, 3])
        
    # Random transformations can be put here: right before you crop images
    # to predefined size. To get more information look at the stackoverflow
    # question linked above.
    image = tf.image.central_crop(image, central_fraction=0.75)
    resized_image = tf.image.resize_images(image, [IMAGE_HEIGHT, IMAGE_WIDTH],
                                           method=tf.image.ResizeMethod.BICUBIC)
    resized_image = tf.saturate_cast(resized_image, dtype=tf.uint8)
    #resized_image = tf.image.resize_image_with_crop_or_pad(image=image,
    #                                       target_height=IMAGE_HEIGHT,
    #                                       target_width=IMAGE_WIDTH)
    
    
    
    images, labels = tf.train.shuffle_batch( [resized_image, label],
                                                 batch_size=batch_size,
                                                 capacity=30,
                                                 num_threads=2,
                                                 min_after_dequeue=10)
    
    return images, labels

filename_queue = tf.train.string_input_producer(
    [tfrecords_filename], num_epochs=10)
batch_size = 4
image, label = read_and_decode(filename_queue, batch_size)

# The op for initializing the variables.
init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())

with tf.Session()  as sess:
    
    sess.run(init_op)
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    
    # Let's read off 3 batches just for example
    for i in range(2):
    
        img, labl = sess.run([image, label])
        print(img[0, :, :, :].shape)
        
        print('current batch')
        
        # We selected the batch size of two
        # So we should get two image pairs in each batch
        # Let's make sure it is random
        for ii in range(batch_size):
            print(labl[ii])
            io.imshow(img[ii, :, :, :]/255.)
            io.show()
    
    coord.request_stop()
    coord.join(threads)
