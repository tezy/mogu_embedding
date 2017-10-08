import os
import csv
import codecs
import random
import numpy as np
import tensorflow as tf
from PIL import Image

from tensorflow.contrib import slim

NUM_CLS = 35
NUM_COLOR = 75
NUM_ATTR = 1160
INPUT_IMAGE_SIZE = 299


class DataSet(object):
    num_cls = 35
    num_clr = 75
    num_attr = 1160

    def __init__(self, dataset_dir, filename):
        self._dataset_dir = dataset_dir
        self._file_record_path = os.path.join(dataset_dir, filename)
        self._image_path = []
        self._image_id_label = []
        self._image_cls_label = []
        self._image_color_label = []
        self._image_attr_label = []
        with codecs.open(self._file_record_path, 'r', 'utf-8') as f:
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
        path = tf.convert_to_tensor(self._image_path, tf.string)
        cls = tf.convert_to_tensor(self._image_cls_label, tf.int32)
        clr = tf.convert_to_tensor(self._image_color_label, tf.int32)

        attr_list = self._process_attr_label()
        attr = tf.convert_to_tensor(attr_list, tf.int32)

        if include_img_id:
            imid = tf.convert_to_tensor(self._image_id_label, tf.int32)
            inputs_info = tf.train.slice_input_producer([path, cls, clr, attr, imid], num_epochs)
        else:
            inputs_info = tf.train.slice_input_producer([path, cls, clr, attr], num_epochs)

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

    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _int64_list_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def _data_shuffle(self):
        attr_array_label = self._process_attr_label()
        dataset_info = list(zip(self._image_path, self._image_id_label, self._image_cls_label,
                                self._image_color_label, attr_array_label))
        random.shuffle(dataset_info)
        return dataset_info

    def create_tfrecords(self):
        tfrecords_name = 'mogu_image_info.tfrecords'
        tfrecords_path = os.path.join(self._dataset_dir, tfrecords_name)
        if os.path.exists(tfrecords_path):
            return
        else:
            shuffled_dataset_info = self._data_shuffle()
            writer = tf.python_io.TFRecordWriter(tfrecords_path)

            for path, id, clz, clr, attr_list in shuffled_dataset_info:
                example = tf.train.Example(features=tf.train.Features(feature={
                    'image_path': self._bytes_feature(path.encode()),
                    'image_id': self._int64_feature(id),
                    'class_label': self._int64_feature(clz),
                    'color_label': self._int64_feature(clr),
                    'attribute_label': self._int64_list_feature(attr_list)
                }))

                writer.write(example.SerializeToString())

            writer.close()

    def input_pipeline_tfrecords(self, batch_size, num_epochs=None, num_threads=4, image_size=299, include_img_id=False):
        tfrecords_name = 'mogu_image_info.tfrecords'
        tfrecords_path = os.path.join(self._dataset_dir, tfrecords_name)
        if not os.path.exists(tfrecords_path):
            self.create_tfrecords()

        filename_queue = tf.train.string_input_producer([tfrecords_path], num_epochs=num_epochs)

        reader = tf.TFRecordReader()

        _, serialized_example = reader.read(filename_queue)
        features = {
            'image_path': tf.FixedLenFeature([], tf.string),
            'image_id': tf.FixedLenFeature([], tf.int64),
            'class_label': tf.FixedLenFeature([], tf.int64),
            'color_label': tf.FixedLenFeature([], tf.int64),
            'attribute_label': tf.FixedLenFeature([self.num_attr], tf.int64)
        }
        inputs_info = tf.parse_single_example(serialized_example, features=features)

        # decode and preprocess the image
        file_content = tf.read_file(inputs_info['image_path'])
        image = tf.image.decode_image(file_content, channels=3)
        image = preprocessing(image, image_size, image_size, channels=3)
        # image = inputs_info['image_path']

        # transform the image_label to one hot encoding
        cls_label = slim.one_hot_encoding(inputs_info['class_label'], self.num_cls)
        clr_label = slim.one_hot_encoding(inputs_info['color_label'], self.num_clr)
        attr_label = inputs_info['attribute_label']

        min_after_dequeue = 100
        capacity = min_after_dequeue + 3*batch_size
        # batching images and labels
        if include_img_id:
            id_label = inputs_info['image_id']
            inputs_batch = tf.train.shuffle_batch([image, id_label, cls_label, clr_label, attr_label],
                                                  batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue,
                                                  num_threads=num_threads)
        else:
            inputs_batch = tf.train.shuffle_batch([image, cls_label, clr_label, attr_label],
                                                  batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue,
                                                  num_threads=num_threads)

        return inputs_batch

    @property
    def num_samples(self):
        return len(self._image_path)

    @property
    def num_image_id(self):
        return self._image_id_label[-1] + 1

#
# def _bytes_feature(value):
#     return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
#
#
# def _int64_feature(value):
#     return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
#
#
# def _int64_list_feature(value):
#     return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


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
    inputs, image_id, clz_label, clr_label, attr_labels = \
        dataset.input_pipeline_tfrecords(batch_size=2, num_epochs=3, include_img_id=True)

    # inputs, cls_labels, clr_labels, attr_labels = inputs_loader(dataset_dir, filename, 1, 20, 299, 299)

    with tf.Session() as sess:
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        # count = 0
        # try:
        #     while not coord.should_stop():
        #         # Run training steps or whatever
        #         image, cls, clr, attr = sess.run([inputs, cls_labels, clr_labels, attr_labels])
        #         print(cls)
        #         print(attr)
        #         print('{:*^20}'.format(''))
        #         print(count)
        #         count += 1
        # except tf.errors.OutOfRangeError:
        #     print('Done training -- epoch limit reached')
        # finally:
        #     # When done, ask the threads to stop.
        #     coord.request_stop()

        # while not coord.should_stop():
        #     # Run training steps or whatever
        #     if count > 10:
        #         coord.request_stop()
        #     image, id, cls, clr, attr = sess.run([inputs, image_id, cls_labels, clr_labels, attr_labels])
        #     print(id)
        #     print(attr)
        #     print('{:*^20}'.format(''))
        #     count += 1

        # Wait for threads to finish.
        # coord.join(threads)
    #
        for batch_idx in range(3):
            img, id, clz, clr, attr = sess.run([inputs, image_id, clz_label, clr_label, attr_labels])
            print('*****batch: {}*****'.format(batch_idx))
            print(img)
            print(clz)
            print(attr)
            for idx, im in enumerate(img):
                pic = decode_image(im)
                pic.save(os.path.join(dataset_dir, 'tf_img_{}.jpg'.format(batch_idx*2+idx)))

        coord.request_stop()
        # Wait for threads to finish.
        coord.join(threads)


if __name__ == '__main__':
    main()