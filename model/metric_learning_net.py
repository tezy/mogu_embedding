import os
import sys
import csv
import re
import random
import codecs
import logging
import argparse
import tensorflow as tf
import numpy as np

from PIL import Image
from tensorflow.contrib import slim
from nets import inception_v4

# image_dataset_dir = '/raid5data/dplearn/taobao/crawler_tbimg/mimages'
#
# data_dir = '/home/deepinsight/tongzhen/data-set/mogu_embedding'
# data_file = 'mogujie_r.json'
#
# save_dir = '/home/deepinsight/tongzhen/data-set/mogu_embedding'
#
# cid_attr_file = 'mogu_category_attrs.json'
# mogu_valid_json = 'mogu_valid.json'
# image_path_file = 'image_path.txt

NUM_CLS = 35
NUM_COLOR = 75
NUM_ATTR = 1160
INPUT_IMAGE_SIZE = 299

TRAIN_SCOPE = ['InceptionV4/Logits', 'InceptionV4/Embeddings', 'InceptionV4/Mixed_7d']


def triplet_loss(anchor, positive, negative, gap):
    """Calculate the triplet loss according to the FaceNet paper

    Args:
      anchor: the embeddings for the anchor images.
      positive: the embeddings for the positive images.
      negative: the embeddings for the negative images.
      gap: the discrimination between positive and negative embedding

    Returns:
      the triplet loss according to the FaceNet paper as a float tensor.
    """
    with tf.variable_scope('triplet_loss'):
        pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)

        basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), gap)
        loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)

    return loss


def center_loss(embedding_batch, update_idx, alpha, num_classes):
    """Center loss based on the paper "A Discriminative Feature Learning Approach for Deep Face Recognition"
       (http://ydwen.github.io/papers/WenECCV16.pdf)
       Args:
           embedding_batch: the batch of the new embeddings
           update_idx: the the centers to be updated
           alpha: centers decay ratio
           num_classes: the overall num classes

       Returns:
           the center loss of the batch
           updated centers
    """
    embedding_dim = embedding_batch.get_shape()[1]
    centers = tf.get_variable('centers', [num_classes, embedding_dim], dtype=tf.float32,
        initializer=tf.constant_initializer(0), trainable=False)
    update_idx = tf.reshape(update_idx, [-1])
    centers_batch = tf.gather(centers, update_idx)
    diff = (1 - alpha) * (centers_batch - embedding_batch)
    centers = tf.scatter_sub(centers, update_idx, diff)
    loss = tf.reduce_mean(tf.square(embedding_batch - centers_batch))

    # the batch central loss and the new centers op according to embedding_batch
    return loss, centers


def image_info_loader(dataset_dir, filename):
    image_path = []
    image_id_label = []
    image_cls_label = []
    image_color_label = []
    image_attr_label = []
    with open(os.path.join(dataset_dir, filename), 'r') as f:
        csv_reader = csv.reader(f)
        for line in csv_reader:
            image_path.append(line[0])
            image_id_label.append(int(line[1]))
            image_cls_label.append(int(line[2]))
            image_color_label.append(int(line[3]))

            # color_label = [int(label) for label in line[3].split()]
            attr_labels = [int(label) for label in line[4].split()]
            attrs = [0] * NUM_ATTR
            for idx in attr_labels:
                attrs[idx] = 1
            image_attr_label.append(attrs)

    return image_path, image_id_label, image_cls_label, image_color_label, image_attr_label


class DataSet(object):
    num_cls = 35
    num_clr = 75
    num_attr = 1160

    def __init__(self, dataset_dir, filename):
        self._file_record_path = os.path.join(dataset_dir, filename)
        self._image_path = []
        self._image_id_label = []
        self._image_cls_label = []
        self._image_color_label = []
        self._image_attr_label = []
        with open(self._file_record_path, 'r') as f:
            csv_reader = csv.reader(f)
            for line in csv_reader:
                self._image_path.append(line[0])
                self._image_id_label.append(int(line[1]))
                self._image_cls_label.append(int(line[2]))
                self._image_color_label.append(int(line[3]))
                self._image_attr_label.append([int(label) for label in line[4].strip().split()])

    def _process_attr_label(self):
        image_attr_label_array = []
        for attr_labels in self._image_attr_label:
            attrs = [0] * self.num_attr
            for idx in attr_labels:
                attrs[idx] = 1
                image_attr_label_array.append(attrs)

        return image_attr_label_array

    def load_inputs(self, batch_size, num_epochs=None, num_threads=4, image_size=299, include_img_id=False):
        # create the filename and label example
        attr_list = self._process_attr_label()
        if include_img_id:
            inputs_info = tf.train.slice_input_producer([self._image_path,
                                                         self._image_cls_label,
                                                         self._image_color_label,
                                                         attr_list,
                                                         self._image_id_label], num_epochs)
        else:
            inputs_info = tf.train.slice_input_producer([self._image_path,
                                                         self._image_cls_label,
                                                         self._image_color_label,
                                                         attr_list], num_epochs)

        # decode and preprocess the image
        file_content = tf.read_file(inputs_info[0])
        image = tf.image.decode_image(file_content, channels=3)
        image = preprocessing(image, image_size, image_size, channels=3)

        # transform the image_label to one hot encoding
        cls_label = slim.one_hot_encoding(inputs_info[1], self.num_cls)
        clr_label = slim.one_hot_encoding(inputs_info[2], self.num_clr)
        attr_label = inputs_info[3]

        # batching images and labels
        if include_img_id:
            id_label = inputs_info[4]
            inputs_batch = tf.train.batch([image, id_label, cls_label, clr_label, attr_label],
                                          batch_size, capacity=5*batch_size, num_threads=num_threads)
        else:
            inputs_batch = tf.train.batch([image, cls_label, clr_label, attr_label],
                                          batch_size, capacity=5*batch_size, num_threads=num_threads)

        return inputs_batch

    @property
    def num_samples(self):
        return len(self._image_path)


def inputs_loader(dataset_dir, filename, batch_size, num_epochs, height, width):
    # create the filename and label example
    path_list, _, cls_list, color_list, attr_list = image_info_loader(dataset_dir, filename)
    path_list = tf.convert_to_tensor(path_list, dtype=tf.string)
    #id_list = tf.convert_to_tensor(id_list, dtype=tf.int32)
    cls_list = tf.convert_to_tensor(cls_list, dtype=tf.int32)
    color_list = tf.convert_to_tensor(color_list, dtype=tf.int32)
    attr_list = tf.convert_to_tensor(attr_list, dtype=tf.int32)

    image_path, cls_label, clr_label, attr_label = \
        tf.train.slice_input_producer([path_list, cls_list, color_list, attr_list], num_epochs)

    # decode and preprocess the image
    file_content = tf.read_file(image_path)
    image = tf.image.decode_image(file_content, channels=3)
    image = preprocessing(image, height, width, channels=3)

    # transform the image_label to one hot encoding
    cls_label = slim.one_hot_encoding(cls_label, NUM_CLS)
    clr_label = slim.one_hot_encoding(clr_label, NUM_COLOR)

    # batching images and labels
    num_threads = 4
    min_after_dequeue = 13 * batch_size
    capacity = min_after_dequeue + 3 * batch_size
    image_batch, cls_label_batch, clr_label_batch, attr_label_batch = \
        tf.train.batch([image, cls_label, clr_label, attr_label], batch_size,
                       capacity=capacity, num_threads=num_threads)

    return image_batch, cls_label_batch, clr_label_batch, attr_label_batch


def get_network_fn(embedding_dims = 128, num_cls=35, num_clr = 75, num_attrs = 1160, weight_decay=0.0004,
                   is_training=True, dropout_keep_prob=0.8, reuse=None, scope='InceptionV4'):

    DEFAULT_SIZE = 299
    def network_fn(inputs):
        arg_scope = inception_v4.inception_v4_arg_scope(weight_decay=weight_decay)
        with slim.arg_scope(arg_scope):
            with tf.variable_scope(scope, 'InceptionV4', [inputs], reuse=reuse) as scope_in:
                with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
                    net, end_points = inception_v4.inception_v4_base(inputs, scope=scope_in)

                    with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                                        stride=1, padding='SAME'):
                        # Final pooling and prediction
                        with tf.variable_scope('Embeddings'):
                            # 8 x 8 x 1536
                            net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID',
                                                  scope='AvgPool_1a')
                            # 1 x 1 x 1536
                            # net = slim.dropout(net, dropout_keep_prob, scope='Dropout_1b')
                            net = slim.flatten(net, scope='PreLogitsFlatten')
                            end_points['PreLogitsFlatten'] = net
                            # 1536 --> 128 embedding dims
                            prelogits = slim.fully_connected(net, embedding_dims, activation_fn=None,
                                                             scope='Prelogits')
                            end_points['Prelogits'] = prelogits

                        with tf.variable_scope('Logits'):
                            cls_logits = slim.fully_connected(prelogits, num_cls, activation_fn=None,
                                                              normalizer_fn=None, normalizer_params=None,
                                                              scope='Classes')
                            end_points['ClassLogits'] = cls_logits
                            clr_logits = slim.fully_connected(prelogits, num_clr, activation_fn=None,
                                                              normalizer_fn=None, normalizer_params=None,
                                                              scope='Colors')
                            end_points['ColorLogits'] = clr_logits
                            attr_logits = slim.fully_connected(prelogits, num_attrs, activation_fn=None,
                                                               normalizer_fn=None, normalizer_params=None,
                                                               scope='Attributes')
                            end_points['AttributeLogits'] = attr_logits

                return cls_logits, clr_logits, attr_logits, prelogits, end_points

    network_fn.default_image_size = DEFAULT_SIZE

    return network_fn


def get_vars_to_train(train_scope):
    vars_to_train = []
    for scope in train_scope:
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        vars_to_train.extend(var_list)

    return vars_to_train


def get_vars_to_restore(exclude_scope):
    exclusions = [scope.strip() for scope in exclude_scope]
    vars_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            vars_to_restore.append(var)

    return vars_to_restore


def preprocessing(image, height, width, channels, augment=False):
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    # some pics are modified by ps software, and extra channels may be added by mistake
    image = tf.squeeze(image)

    image = tf.image.central_crop(image, central_fraction=0.975)
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, [height, width], align_corners=False)
    image = tf.squeeze(image, [0])
    image = tf.image.random_flip_left_right(image)
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    image.set_shape((height, width, channels))
    return image


def preprocess_for_eval(image, height, width,
                        central_fraction=0.875, scope=None):
  with tf.name_scope(scope, 'eval_image', [image, height, width]):
    if image.dtype != tf.float32:
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    # Crop the central region of the image with an area containing 87.5% of
    # the original image.
    if central_fraction:
      image = tf.image.central_crop(image, central_fraction=central_fraction)

    if height and width:
      # Resize the image to the specified height and width.
      image = tf.expand_dims(image, 0)
      image = tf.image.resize_bilinear(image, [height, width],
                                       align_corners=False)
      image = tf.squeeze(image, [0])
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    return image


def decode_image(file_content):
    image = file_content / 2.0
    image = image + 0.5
    image = image * 255.0
    return Image.fromarray(image.astype(np.uint8))


def main():
    # inputs = tf.placeholder(dtype=tf.float32, shape=[None, 299, 299, 3])
    # network_fn = get_network_fn()
    # cls_logits, clr_logits, attr_logits, prelogits, _ = network_fn(inputs)
    #
    # for var in tf.get_collection(tf.GraphKeys.MODEL_VARIABLES):
    #     print(var)
    dataset_dir = '/home/tze/Learning/dataset/mogu_embedding'
    filename = 'sample_local.csv'
    dataset = DataSet(dataset_dir, filename)
    inputs, cls_labels, clr_labels, attr_labels = dataset.load_inputs(8, 20, 299)

    # inputs, cls_labels, clr_labels, attr_labels = inputs_loader(dataset_dir, filename, 1, 20, 299, 299)

    with tf.Session() as sess:
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        count = 0
        try:
            while not coord.should_stop():
                # Run training steps or whatever
                image, cls, clr, attr = sess.run([inputs, cls_labels, clr_labels, attr_labels])
                print(count)
                count += 1
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()

        # Wait for threads to finish.
        coord.join(threads)


if __name__ == '__main__':
    main()












