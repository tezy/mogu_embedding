import os
import sys

root_path = os.path.abspath('..')
sys.path.append(root_path)

import tensorflow as tf
from data import dataset_loader



filename_dir = '/home/tze/Learning/dataset/mogu_embedding'
filename = 'sample_local.csv'
# filename_dir = '/home/tze/Tmp/mogu_embedding/data'
# filename = 'mogu_train_fin_tze_least_3.csv'
save_dir = '/home/tze/Learning/vars/mogu_embedding'
ckpt_dir = '/home/tze/Learning/ckpt/standard/inception_v4.ckpt'

file_path = ['/home/tze/App', '/home/tze/Workspace', '/home/tze/Tmp', '/home/tze/Learning', '/home/tze/Downloads']
clz_label = [3, 5, 7, 10, 24]
attr_label = [[0, 5, 7],
              [5, 7, 3],
              [0, 1, 0],
              [1, 1, 1],
              [0, 0, 1]]


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def tfrecords_store():
    tfrecords_name = 'tfrecords_test.tfrecords'
    tfrecords_path = os.path.join(filename_dir, tfrecords_name)
    if os.path.exists(tfrecords_path):
        return
    else:
        inputs_info = list(zip(file_path, clz_label, attr_label))
        writer = tf.python_io.TFRecordWriter(tfrecords_path)

        for path, clz, attr_list in inputs_info:
            features = tf.train.Features(feature={
                'file_path': _bytes_feature(path.encode()),
                'class_label': _int64_feature(clz),
                'attribute_label': _int64_list_feature(attr_list)})
            example = tf.train.Example(features=features)

            writer.write(example.SerializeToString())

        writer.close()


def tfrecords_reader(batch_size, num_epochs):
    tfrecords_name = 'tfrecords_test.tfrecords'
    tfrecords_path = os.path.join(filename_dir, tfrecords_name)
    filename_queue = tf.train.string_input_producer([tfrecords_path], num_epochs=None)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = {
        'file_path': tf.FixedLenFeature([], tf.string),
        'class_label': tf.FixedLenFeature([], tf.int64),
        'attribute_label': tf.FixedLenFeature([3], tf.int64)
    }
    inputs_info = tf.parse_single_example(serialized_example, features=features)

    # decode and preprocess the image
    # file_content = tf.decode_raw(inputs_info['image_path'])

    # transform the image_label to one hot encoding
    file_path = inputs_info['file_path']
    cls_label = inputs_info['class_label']
    attr_label = inputs_info['attribute_label']

    min_after_dequeue = 10
    capacity = min_after_dequeue + batch_size
    # batching images and labels
    inputs_batch = tf.train.shuffle_batch([file_path, cls_label, attr_label], batch_size,
                                          capacity=capacity, min_after_dequeue=min_after_dequeue, num_threads=4)

    return inputs_batch
    # return file_path, cls_label, attr_label


def main():
    # dataset = dataset_loader.DataSet(filename_dir, filename)
    # dataset.create_tfrecords()

    tfrecords_store()
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    file_path, clz_label, attr_label = tfrecords_reader(batch_size=2, num_epochs=10)

    with tf.Session() as sess:
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for batch_idx in range(5):
            path, clz, attr = sess.run([file_path, clz_label, attr_label])
            print('*****batch: {}*****'.format(batch_idx))
            print(path)
            print(clz)
            print(attr)

        coord.request_stop()
        # Wait for threads to finish.
        coord.join(threads)

        # batch_idx = 0
        # try:
        #     while not coord.should_stop():
        #         # Run training steps or whatever
        #         path, clz, attr = sess.run([file_path, clz_label, attr_label])
        #         print('*****batch: {}*****'.format(batch_idx))
        #         print(path)
        #         print(clz)
        #         print(attr)
        #         batch_idx += 1
        # except tf.errors.OutOfRangeError:
        #     print('Done training -- epoch limit reached')
        # finally:
        #     # When done, ask the threads to stop.
        #     coord.request_stop()
        #
        # coord.join(threads)



if __name__ == '__main__':
    main()