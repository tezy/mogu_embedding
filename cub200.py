import os
import csv
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from PIL import Image


IMAGE_ID_FILE = 'images.txt'
IMAGE_ID_LABEL = 'image_class_labels.txt'
TRAIN_TEST_SPLIT = 'train_test_split.txt'
IMAGE_FILE_DIR = 'images'
NUM_CLASSES = 200


class Cub200Loader(object):
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.image_for_train, self.label_for_train, self.image_for_test, self.label_for_test = \
            _get_train_test_set(dataset_dir)
        self.num_sample_for_train = len(self.image_for_train)
        self.num_sample_for_eval = len(self.image_for_test)

    def get_num_sample_for_train(self):
        return self.num_sample_for_train

    @property
    def num_samples(self):
        return self.num_sample_for_train

    def get_num_sample_for_eval(self):
        return self.num_sample_for_eval

    def input_pipeline(self, num_epoch, batch_size, height, width, is_training=True):
        """Get the cub data input pipeline for the graph to train

        """
        if is_training:
            images = tf.convert_to_tensor(self.image_for_train, dtype=tf.string)
            labels = tf.convert_to_tensor(self.label_for_train, dtype=tf.int64)
        else:
            images = tf.convert_to_tensor(self.image_for_test, dtype=tf.string)
            labels = tf.convert_to_tensor(self.label_for_test, dtype=tf.int64)

        image_queue, label_queue = tf.train.slice_input_producer([images, labels], num_epoch, shuffle=True)

        file_content = tf.read_file(image_queue)
        # reader = tf.WholeFileReader()
        # _, file_content = reader.read(image_queue)
        image = tf.image.decode_image(file_content, channels=3)
        image = preprocess(image, height, width)
        label = slim.one_hot_encoding(label_queue, NUM_CLASSES)

        num_threads = 4
        min_after_dequeue = 13 * batch_size
        capacity = min_after_dequeue + 3 * batch_size
        image_batch, label_batch = tf.train.batch([image, label], batch_size,
                                                  num_threads=num_threads, capacity=capacity)

        return image_batch, label_batch


def input_pipeline(dataset_dir, num_epoch, batch_size, height, width, num_classes, is_training=True):
    """Get the cub data input pipeline for the graph to train

    """
    image_for_train, label_for_train, image_for_test, label_for_test = _get_train_test_set(dataset_dir)
    if is_training:
        images = tf.convert_to_tensor(image_for_train, dtype=tf.string)
        labels = tf.convert_to_tensor(label_for_train, dtype=tf.int64)
    else:
        images = tf.convert_to_tensor(image_for_test, dtype=tf.string)
        labels = tf.convert_to_tensor(label_for_test, dtype=tf.int64)

    image_queue, label_queue = tf.train.slice_input_producer([images, labels], num_epoch, shuffle=True)

    file_content = tf.read_file(image_queue)
    image = tf.image.decode_image(file_content, channels=3)
    image = preprocess(image, height, width, is_training)
    label = slim.one_hot_encoding(label_queue, num_classes)

    num_threads = 4
    min_after_dequeue = 7 * batch_size
    capacity = min_after_dequeue + 3 * batch_size
    image_batch, label_batch = tf.train.batch([image, label], batch_size,
                                              num_threads=num_threads, capacity=capacity)

    return image_batch, label_batch


def preprocess(image, height, width, central_fraction=0.875):
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.central_crop(image, central_fraction=central_fraction)
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, [height, width], align_corners=False)
    image = tf.squeeze(image, [0])
    image = tf.image.random_flip_left_right(image)
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    image.set_shape((height, width, 3))
    return image


def create_train_test_file(dataset_dir, save_dir):
    image_for_train, label_for_train, image_for_test, label_for_test = _get_train_test_set(dataset_dir)

    with open(os.path.join(save_dir, 'image_for_train.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerows(zip(image_for_train, label_for_train))

    with open(os.path.join(save_dir, 'image_for_eval.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerows(zip(image_for_test, label_for_test))


def _get_train_test_set(dataset_dir):
    image_file_path, image_class_label = _get_image_file_label(dataset_dir)
    image_for_train = []
    label_for_train = []
    image_for_test = []
    label_for_test = []
    with open(os.path.join(dataset_dir, TRAIN_TEST_SPLIT)) as f:
        for line in f:
            image_id, is_training = line.split()
            image_id = int(image_id) - 1
            is_training = True if int(is_training) == 1 else False

            if is_training:
                image_for_train.append(image_file_path[image_id])
                label_for_train.append(image_class_label[image_id])
            else:
                image_for_test.append(image_file_path[image_id])
                label_for_test.append(image_class_label[image_id])

    return image_for_train, label_for_train, image_for_test, label_for_test


def _get_image_file_label(dataset_dir):
    image_file_path = []
    with open(os.path.join(dataset_dir, IMAGE_ID_FILE), 'r') as f:
        for line in f:
            image_id, image_file_name = line.split()
            image_file_path.append(os.path.join(dataset_dir, IMAGE_FILE_DIR, image_file_name))

    image_class_label = []
    with open(os.path.join(dataset_dir, IMAGE_ID_LABEL), 'r') as f:
        for line in f:
            image_id, image_label = line.split()
            image_class_label.append(int(image_label) - 1)

    return image_file_path, image_class_label


def _write_to_document(file_name, src):
    with open(file_name, 'w') as f:
        for item in src:
            f.write(str(item) + '\n')


def main():
    dataset_dir = '/home/tze/Workspace/data-set/cub200/CUB_200_2011'
    save_dir = '/home/tze'
    create_train_test_file(dataset_dir, save_dir)
    # image_file_path, image_class_label = _get_image_file_label(dataset_dir)
    # image_for_train, label_for_train, image_for_test, label_for_test = _get_train_test_set(dataset_dir)

    # _write_to_document('/home/tze/image_path', image_file_path)
    # _write_to_document('/home/tze/image_label', image_class_label)
    # _write_to_document('/home/tze/train_image', image_for_train)
    # _write_to_document('/home/tze/train_label', label_for_train)
    # _write_to_document('/home/tze/test_image', image_for_test)
    # _write_to_document('/home/tze/test_label', label_for_test)

    # batch_size = 32
    # cub_data_set = Cub200Loader(dataset_dir)
    # print(cub_data_set.get_num_sample_for_train())
    # image_batch, label_batch = cub_data_set.input_pipeline(2, batch_size, 224, 224)
    #
    # # image_batch, label_batch = input_pipeline(dataset_dir, 2, batch_size, 224, 224, 200, is_training=True)
    #
    # init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    #
    # sess = tf.Session()
    #
    # # Initialize the variables (like the epoch counter).
    # sess.run(init_op)
    #
    # # Start input enqueue threads.
    # coord = tf.train.Coordinator()
    # threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    #
    # try:
    #     for i in range(4):
    #         print(i)
    #         print(sess.run(label_batch))
    #         images = sess.run(image_batch)
    #         print(images.shape)
    #         print()
    #         for j, image in enumerate(images):
    #             im = Image.fromarray((image * 255).astype(np.uint8))
    #             im.save('/home/tze/Tmp/image/image_{}.jpg'.format(i* batch_size + j))
    # except tf.errors.OutOfRangeError:
    #     print('Done training -- epoch limit reached')
    # finally:
    #     # When done, ask the threads to stop.
    #     coord.request_stop()
    #
    # # Wait for threads to finish.
    # coord.join(threads)
    # sess.close()


if __name__ == '__main__':
    main()

