import time
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim

from datetime import datetime
from model import metric_learning_net
from data import dataset_loader

weight_decay = 4e-5
label_smoothing = 0.1
rmsprop_decay = 0.9
num_epochs_per_decay = 2
init_learning_rate = 0.01
end_learning_rate = 0.00001
learning_rate_decay_factor = 0.94
dropout_keep_prob = 1.0


def train_loss(cls_labels, clr_labels, attr_labels, cls_logits, clr_logits, attr_logits):
    # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits,
    #                                                                name='cross_entropy_per_example')
    # avg_cross_entropy = tf.reduce_mean(cross_entropy)
    tf.losses.softmax_cross_entropy(onehot_labels=cls_labels, logits=cls_logits,
                                               label_smoothing=label_smoothing)
    tf.losses.softmax_cross_entropy(onehot_labels=clr_labels, logits=clr_logits,
                                                label_smoothing=label_smoothing)
    tf.losses.sigmoid_cross_entropy(multi_class_labels=attr_labels, logits=attr_logits,
                                                label_smoothing=label_smoothing)
    label_losses = tf.get_collection(tf.GraphKeys.LOSSES)
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

    return tf.add_n(label_losses + regularization_losses, name='total_loss')


# def train_accuracy(labels, logits):
#     correct_prediction = tf.equal(tf.argmax(logits, axis=1), tf.argmax(labels, axis=1))
#     with tf.name_scope('accuracy'):
#         accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#     tf.summary.scalar('accuracy', accuracy)
#
#     return accuracy


def train_step(total_loss, global_step, train_scope, num_samples_per_epoch, batch_size):
    decay_step = int(num_samples_per_epoch / batch_size * num_epochs_per_decay)
    lr = tf.train.exponential_decay(init_learning_rate, global_step, decay_step,
                                    learning_rate_decay_factor, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(lr)


    # ema_loss = tf.train.ExponentialMovingAverage(0.9, name='avg')
    # ema_loss_op = ema_loss.apply([total_loss])
    tf.summary.scalar(total_loss.op.name + '(raw)', total_loss)
    # tf.summary.scalar(total_loss.op.name, ema_loss.average(total_loss))

    vars_to_train = get_vars_to_train(train_scope)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # update_ops.append(ema_loss_op)
    mini_loss_op = optimizer.minimize(total_loss, global_step, var_list=vars_to_train)
    with tf.control_dependencies(update_ops):
        with tf.control_dependencies([mini_loss_op]):
            train_op = tf.no_op('train_step')

    return train_op


def init_fn(sess, fine_tune_ckpt, save_dir, exclude_scope, is_fine_tuning=True):
    """Initialize the variables for fine tuning and training

    """
    sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

    if is_fine_tuning:
        # initialize the vars for fine turning
        fine_tune_var_list = get_vars_to_restore(exclude_scope)
        saver = tf.train.Saver(fine_tune_var_list)
        saver.restore(sess, fine_tune_ckpt)
        print('load the fine tune model successfully')
    else:
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print(ckpt.model_checkpoint_path)
            print('load the ckpt file successfully')
            sess.run(tf.train.get_global_step().assign(0))

    # sess.run(tf.variables_initializer(
    #     list(tf.get_variable(name) for name in sess.run(tf.report_uninitialized_variables()))
    # ))


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


def main():
    filename_dir = '/home/tze/Learning/dataset/mogu_embedding'
    filename = 'sample_local.csv'
    save_dir = '/home/tze/Learning/vars/mogu_embedding'
    fine_tune_ckpt = '/home/tze/Learning/ckpt/standard/inception_v4.ckpt'

    train_scopes = ['InceptionV4/Logits', 'InceptionV4/Embeddings', 'InceptionV4/Mixed_7d']
    exclude_scopes = train_scopes

    batch_size = 16
    num_epochs = 20
    image_size = 299
    dataset = dataset_loader.DataSet(filename_dir, filename)
    images, cls_labels, clr_labels, attr_labels = dataset.load_inputs(batch_size, num_epochs)

    global_step = slim.create_global_step()

    network_fn = metric_learning_net.get_network_fn(weight_decay=weight_decay)
    cls_logits, clr_logits, attr_logits, prelogits, _ = network_fn(images)

    batch_loss = train_loss(cls_labels, clr_labels, attr_labels, cls_logits, clr_logits, attr_logits)

    train_op = train_step(batch_loss, global_step, train_scopes, dataset.num_samples, batch_size)

    # summary_op = tf.summary.merge_all()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        init_fn(sess, fine_tune_ckpt, save_dir, exclude_scopes)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            while not coord.should_stop():
                # Run training steps or whatever
                loss, step, _ = sess.run([batch_loss, global_step, train_op])
                if step % 777 == 0:
                    print('step: {:>5}, loss: {:.5f}'.format(step, loss))
                    saver.save(sess, os.path.join(save_dir, 'mogu.ckpt'), global_step=global_step)
                    print('save model ckpt')
                else :
                    print('step: {:>5}, loss: {:.5f}'.format(step, loss))

        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()

        # Wait for threads to finish.
        coord.join(threads)


if __name__ == '__main__':
    main()
