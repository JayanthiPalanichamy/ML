#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 09:49:56 2017

@author: vijay
"""

import tensorflow as tf
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from tensorflow.python.platform import tf_logging as logging
from inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope
from tensorflow.contrib.layers.python.layers import initializers
import os
import time
import matplotlib.pyplot as plt
import numpy as np

slim = tf.contrib.slim
tf.logging.set_verbosity(tf.logging.INFO)

tf.app.flags.DEFINE_string('checkpoint_file', 
                           '/home/vijay/datasets/pre_trained_models/inception_resnet_v2_2016_08_30.ckpt',
                           'String: Inception resnet v2 model file.')
tf.app.flags.DEFINE_string('dataset_dir',
                           '/home/vijay/datasets/flowers_data/',
                           'String: TFRecord directory.')
tf.app.flags.DEFINE_string('train_file_pattern',
                           'flowers_train_',
                           'String: File pattern for train.')
tf.app.flags.DEFINE_string('test_file_pattern',
                           'flowers_validation_',
                           'String: File pattern for train.')
tf.app.flags.DEFINE_integer('num_classes', 5,
                            'Int: Number of classes.')
tf.app.flags.DEFINE_integer('image_size', 299,
                            'Int: The model expected image size.')
tf.app.flags.DEFINE_integer('num_preprocess_threads', 4,
                            """Int: Number of preprocessing threads per tower. """
                            """Please make this a multiple of 4.""")
tf.app.flags.DEFINE_integer('batch_size', 30,
                            'Int: Number of images to process in a batch.')
tf.app.flags.DEFINE_integer('input_queue_memory_factor', 4,
                            'Int: size of the queue of preprocessed images. ')
tf.app.flags.DEFINE_integer('num_readers', 4,
                            'Int: Number of parallel readers during train.')
tf.app.flags.DEFINE_string('log_dir',
                           '/home/vijay/logs/flower',
                           'String: The model to be stored')
FLAGS = tf.app.flags.FLAGS

# Learning rate hyperparameters
num_epochs = 50
num_epochs_before_decay = 2
initial_learning_rate = 0.001
learning_rate_decay_factor = 0.16

def plot_prod_images(images, labels, rows=8):
    fig, axes = plt.subplots(rows,int(np.ceil(len(images)/rows)))
    #fig.subplots_adjust(hspace=0.3, wspace=0.3)
    
    for i, (image, ax) in enumerate(zip(images, axes.flat)):
        
        ax.imshow((image+1.0)/2.0)
        ax.set_xlabel(labels[i])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
    plt.suptitle('Train Images')
    plt.show(1)

def decode_jpeg(image_buffer):
    try:
        image = tf.image.decode_jpeg(image_buffer, channels=3)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    except Exception as e:
        logging.warning('error occured: {}'.format(e))
        return None
    return image

def distort_image(image, height, width, thread_id=0):
    
    image = tf.image.central_crop(image, central_fraction=0.85)
    
    # cropped image with bboxes randomly distorted
    
#    bbox = tf.constant([0.2, 0.2, 0.6, 0.8],
#                         dtype=tf.float32,
#                         shape=[1, 1, 4])
#    begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(
#            tf.shape(image), 
#            bounding_boxes=bbox,
#            aspect_ratio_range=[0.75,1],
#            area_range=[0.5, 1])
#    image = tf.slice(image, begin, size)
    
    resize_method = thread_id % 4
    distorted_image = tf.image.resize_images(image, [height, width],
                                             method=resize_method)
    
    distorted_image.set_shape([height, width, 3])
    
    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    
    return distorted_image
    
def eval_image(image, height, width):
    
    image = tf.image.central_crop(image, central_fraction=0.875)

    # Resize the image to the original height and width.
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, [height, width],
                                     align_corners=False)
    image = tf.squeeze(image, [0])
    return image
    
def image_transform(image_buffer, train, thread_id=0):
    """ Decode and preprocess one image for evaluation or training.
    """
    image = decode_jpeg(image_buffer)
    if image is None:
        return None
    height = FLAGS.image_size
    width = FLAGS.image_size
    
    if train:
        image = distort_image(image, height, width, thread_id)
    else:
        image = eval_image(image, height, width)
    
    # Finally, rescale to [-1,1] instead of [0, 1)
    image = tf.scalar_mul((1.0/255), image)
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    return image
    

def parse_example_proto(example_serialized):
    """ Parses the example proto containing the training of the image.
    """
    feature_map = {
      'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                          default_value=''),
      'image/class/label': tf.FixedLenFeature([1], dtype=tf.int64,
                                              default_value=-1),
      'image/class/text': tf.FixedLenFeature([], dtype=tf.string,
                                             default_value='')}
    features = tf.parse_single_example(example_serialized, feature_map)
    label = tf.cast(features['image/class/label'], dtype=tf.int32)
    
    return features['image/encoded'], label

def image_processing_module(dataset, train):
    
    """Contruct batches of training or evaluation examples from the image dataset.
    """
    
    batch_size = FLAGS.batch_size
    num_readers = FLAGS.num_readers
    num_preprocess_threads = FLAGS.num_preprocess_threads
    
    with tf.device('/cpu:0'):
        with tf.name_scope('batch_processing'):
            if train:
                filename_queue = tf.train.string_input_producer(dataset,
                                                                shuffle=True,
                                                                capacity=2)
            else:
                filename_queue = tf.train.string_input_producer(dataset,
                                                                shuffle=False,
                                                                capacity=1)
            examples_per_shard = 128
            # Shuffling queue size
            min_queue_examples = examples_per_shard * FLAGS.input_queue_memory_factor
            
            if train:
                 examples_queue = tf.RandomShuffleQueue(
                         capacity=min_queue_examples + 3 * batch_size,
                         min_after_dequeue=min_queue_examples, 
                         dtypes=[tf.string])
            else:
                 examples_queue = tf.FIFOQueue(
                         capacity=examples_per_shard + 3 * batch_size,
                         dtypes=[tf.string])
                 
            # Create multiple readers to populate the queue of examples.
            if num_readers > 1:
                enqueue_ops = []
                for _ in range(num_readers):
                    reader = tf.TFRecordReader()
                    _, value = reader.read(filename_queue)
                    enqueue_ops.append(examples_queue.enqueue([value]))

                tf.train.queue_runner.add_queue_runner(
                tf.train.queue_runner.QueueRunner(examples_queue, enqueue_ops))
                example_serialized = examples_queue.dequeue()
            else:
                reader = tf.TFRecordReader()
                _, example_serialized = reader.read(filename_queue)
            
            images_and_labels = []
            
            for thread_id in range(num_preprocess_threads):
                # Parse a serialized Examle proto to extract the image and metadata.
                image_buffer, labels = parse_example_proto(example_serialized)
                
                image = image_transform(image_buffer, train, thread_id)
                if image is not None:
                    images_and_labels.append([image, labels])
                
            images, label_index_batch = tf.train.batch_join(
                    images_and_labels, batch_size=batch_size, 
                    capacity=2*num_preprocess_threads*batch_size)
            
            #Reshape images into desired dimentions
            height = FLAGS.image_size
            width = FLAGS.image_size
            depth = 3
            
            images = tf.cast(images, tf.float32)
            images = tf.reshape(images, shape=[batch_size, height, width, depth])
            
            return images, tf.reshape(label_index_batch, [batch_size])
            
def train(dataset, num_samples):
    
    num_classes = FLAGS.num_classes
    
    assert tf.gfile.Exists(FLAGS.checkpoint_file)
    
    with tf.Graph().as_default():
        
        
        # Loads data batches
        images, labels = image_processing_module(dataset, train=True)
        
        # number of steps
        num_batches_per_epoch =  num_samples//FLAGS.batch_size
        decay_steps = int(num_epochs_before_decay * num_batches_per_epoch)

        #Create the model inference
        with slim.arg_scope(inception_resnet_v2_arg_scope()):
            logits, end_points = inception_resnet_v2(images, num_classes = num_classes, is_training = False)
        
        #Define the scopes to exclude
        #exclude = ['InceptionResnetV2/AuxLogits']
        exclude = ['InceptionResnetV2/Logits', 'InceptionResnetV2/AuxLogits']
        variables_to_restore = slim.get_variables_to_restore(exclude = exclude)
        
        
        #Perform one-hot-encoding of the labels 
        one_hot_labels = slim.one_hot_encoding(labels, num_classes)
        
        #add logits
        #add_logits = slim.fully_connected(end_points['PreLogitsFlatten'] , num_classes, 
        #                                  activation_fn=None, 
        #                                  weights_initializer=tf.truncated_normal_initializer(stddev=0.001))
        
        #apply softmax
        loss = tf.losses.softmax_cross_entropy(one_hot_labels, logits)
        
        #Get the regularization losses as well
        total_loss = tf.losses.get_total_loss()
        
        #Create the global step for monitoring the learning_rate and training.
        global_step = get_or_create_global_step()
        
        #Define the exponential learning rate
        lr = tf.train.exponential_decay(
            learning_rate = initial_learning_rate,
            global_step = global_step,
            decay_steps = decay_steps,
            decay_rate = learning_rate_decay_factor,
            staircase = True)
        
        #Define the optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        
        #Create the train_op
        train_op = slim.learning.create_train_op(total_loss, optimizer)
        
        #accuracy
        
        #probabilities = tf.nn.softmax(tf.concat([logits, add_logits], axis=1), name='new_predictions')

        #predictions = tf.argmax(tf.nn.softmax(add_logits),1)
        predictions = tf.argmax(end_points['Predictions'],1)
        probabilities = end_points['Predictions']
        accuracy, accuracy_update = tf.contrib.metrics.streaming_accuracy(predictions, labels)
        metrics_op = tf.group(accuracy_update, probabilities)
        
        # Create summaries to monitor
        tf.summary.scalar('losses/Total_Loss', total_loss)
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('learning_rate', lr)
        my_summary_op = tf.summary.merge_all()
        
        #training step function
        def train_step(sess, train_op, global_step):
            '''
            Simply runs a session for the three arguments provided and gives a logging on the time elapsed for each global step
            '''
            #Check the time for each sess run
            #start_time = time.time()
            total_loss, global_step_count, _ = sess.run([train_op, global_step, metrics_op])
            #time_elapsed = time.time() - start_time

            #Run the logging to print some results
            #logging.info('global step %s: loss: %.4f ', global_step_count, total_loss)

            return total_loss, global_step_count
        
        #Now we create a saver function that actually restores the variables from a checkpoint file in a sess
        saver = tf.train.Saver(variables_to_restore)
        def restore_fn(sess):
            return saver.restore(sess, FLAGS.checkpoint_file)
        
        #define a supervisor to run a managed session
        sv = tf.train.Supervisor(logdir = FLAGS.log_dir, summary_op = None, init_fn = restore_fn)

        with sv.managed_session() as sess:
            
            for step in range(num_batches_per_epoch*num_epochs):
                if step % (num_batches_per_epoch) == 0:
                    logging.info('Epoch %s/%s', step/num_batches_per_epoch + 1, num_epochs)
                    learning_rate_value, accuracy_value = sess.run([lr, accuracy])
                    logging.info('Current Learning Rate: %s', learning_rate_value)
                    logging.info('Current Streaming Accuracy: %s', accuracy_value)
                    
                    logits_value, probabilities_value, predictions_value, labels_value = sess.run([logits, probabilities, predictions, labels])
                    #print( 'logits: \n', logits_value)
                    #print( 'Probabilities: \n', probabilities_value)
                    print('predictions: \n', predictions_value)
                    print( 'Labels:\n', labels_value)
				   
                #Log the summaries every 10 step.
                if step % 100 == 0:
                    loss, _ = train_step(sess, train_op, sv.global_step)
                    summaries = sess.run(my_summary_op)
                    sv.summary_computed(sess, summaries)
                    logging.info('global step %s: loss: %.4f ', sv.global_step, loss)
                
                #If not, simply run the training step
                else:
                    loss, _ = train_step(sess, train_op, sv.global_step)
                    
            #We log the final training loss and accuracy
            logging.info('Final Loss: %s', loss)
            logging.info('Final Accuracy: %s', sess.run(accuracy))
            
            #Once all the training has been done, save the log files and checkpoint model
            logging.info('Finished training! Saving model to disk now.')
            sv.saver.save(sess, sv.save_path, global_step = sv.global_step)
            
        
# Read the record filenames
def load_dataset(data_dir, train_file_pattern):
    
    num_samples = 0
    tf_record_pattern = os.path.join(data_dir, '%s*' %train_file_pattern)
    logging.info('Loading data from: %s', tf_record_pattern)
    dataset = tf.gfile.Glob(tf_record_pattern)
    
    # Count the total number of files in all the shard
    for tfrecord_file in dataset:
        for record in tf.python_io.tf_record_iterator(tfrecord_file):
            num_samples += 1
    
    
    return dataset, num_samples

    

def main(_):
    
    
    tf.reset_default_graph()
    
    #Print the configuration
    
    dataset, num_samples = load_dataset(FLAGS.dataset_dir, FLAGS.train_file_pattern) 
    logging.info('Number of samples loaded: %d', num_samples)
    if dataset is None:
        logging.info('Dataset files not found in this dataset')
        return None
    logging.info('Dataset Processed..Training started...')
    train(dataset, num_samples)

if __name__ == '__main__':
    tf.app.run()


