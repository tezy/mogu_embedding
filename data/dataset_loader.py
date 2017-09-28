import os
import csv
import tensorflow as tf

from tensorflow.contrib import slim

NUM_CLS = 35
NUM_COLOR = 75
NUM_ATTR = 1160
INPUT_IMAGE_SIZE = 299


class DataSet(object):
    num_id = 626
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

    @property
    def num_samples(self):
        return len(self._image_path)


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