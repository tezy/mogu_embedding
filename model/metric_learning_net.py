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
NUM_CLR = 75
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
           alpha: centers decay ratio, the ratio of previous center values
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


def lables_from_embedding(inputs, embedding_dims=128, weight_decay=0.00004, is_training=True,
                          reuse=None, scope='InceptionV4'):

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
                        #net = slim.dropout(net, dropout_keep_prob, scope='Dropout_1b')
                        net = slim.flatten(net, scope='PreLogitsFlatten')
                        end_points['PreLogitsFlatten'] = net
                        # 1536 --> embedding dims
                        prelogits = slim.fully_connected(net, embedding_dims, activation_fn=None,
                                                         scope='Prelogits')
                        end_points['Prelogits'] = prelogits

                    with tf.variable_scope('Logits'):
                        cls_logits = slim.fully_connected(prelogits, NUM_CLS, scope='FC_Classes_0')
                        cls_logits = slim.fully_connected(cls_logits, NUM_CLS, activation_fn=None,
                                                          normalizer_fn=None, normalizer_params=None,
                                                          scope='FC_Classes_1')
                        end_points['ClassLogits'] = cls_logits

                        clr_logits = slim.fully_connected(prelogits, NUM_CLR, scope='FC_Colors_0')
                        clr_logits = slim.fully_connected(clr_logits, NUM_CLR, activation_fn=None,
                                                          normalizer_fn=None, normalizer_params=None,
                                                          scope='FC_Colors_1')
                        end_points['ColorLogits'] = clr_logits

                        attr_logits = slim.fully_connected(prelogits, NUM_ATTR, activation_fn=None,scope='FC_Attributes_0')
                        attr_logits = slim.fully_connected(attr_logits, NUM_ATTR, activation_fn=None,
                                                           normalizer_fn=None, normalizer_params=None,
                                                           scope='FC_Attributes_1')
                        end_points['AttributeLogits'] = attr_logits

            return cls_logits, clr_logits, attr_logits, prelogits, end_points


def get_network_fn(embedding_dims=128, network_model='labels_from_embedding',
                   weight_decay=0.00004, is_training=True):

    DEFAULT_SIZE = 299
    network_map = {'labels_from_embedding': lables_from_embedding}
    fn = network_map[network_model]

    def network_fn(inputs):
        return fn(inputs, embedding_dims, weight_decay, is_training)

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


# def test():
#    pass
#
# if __name__ == '__main__':
#     main()












