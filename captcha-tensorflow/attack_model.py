# -*- coding:utf-8 -*-
import PIL
import argparse
import datetime
import sys
import tensorflow as tf
import numpy as np
import datasets.base as input_data
import  os
import matplotlib.pyplot as plt
# os.environ["CUDA_VISIBLE_DEVICES"]='2,3'
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
MAX_STEPS = 10000
BATCH_SIZE = 50
TEST_STEPS = 100
class_num=62
c_dir = os.getcwd() + '/captcha-tensorflow'

LOG_DIR = 'log/cnn1-run-%s' % datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
FLAGS = None
#
def save(sess,):
    saver = tf.train.Saver()
    saver.save(sess, "model/"+FLAGS.data_dir[7:-1]+"/model.ckpt")


def restore(sess):
    saver = tf.train.Saver()
    saver.restore(sess, c_dir+"/model/"+FLAGS.data_dir[7:-1]+"/model.ckpt")

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        # with tf.name_scope('stddev'):
        #    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        # tf.summary.scalar('stddev', stddev)
        # tf.summary.scalar('max', tf.reduce_max(var))
        # tf.summary.scalar('min', tf.reduce_min(var))
        # tf.summary.histogram('histogram', var)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

def step_target_class_adversarial_images(x,logits, one_hot_target_class,eps=0.8):
    cross_entropy = tf.losses.softmax_cross_entropy(one_hot_target_class,
                                                    logits,
                                                    label_smoothing=0.1,
                                                    weights=1.0)
    x_adv = x - eps * tf.sign(tf.gradients(cross_entropy, x)[0])

    x_adv = tf.clip_by_value(x_adv, 0, 255)

    return tf.stop_gradient(x_adv)


def stepll_adversarial_images(x,logits):

    least_likely_class = tf.argmin(logits, 1)
    one_hot_ll_class = tf.one_hot(least_likely_class, class_num)
    return step_target_class_adversarial_images(x, logits, one_hot_ll_class)


def main(_):
    # load data
    print('开始')
    meta, train_data, test_data = input_data.load_data(c_dir+'/'+FLAGS.data_dir, flatten=False)
    print('data loaded')
    print('train images: %s. test images: %s' % (train_data.images.shape[0], test_data.images.shape[0]))

    LABEL_SIZE = meta['label_size']
    NUM_PER_IMAGE = meta['num_per_image']
    IMAGE_HEIGHT = meta['height']
    IMAGE_WIDTH = meta['width']
    IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT
    print('label_size: %s, image_size: %s' % (LABEL_SIZE, IMAGE_SIZE))

    # variable in the graph for input data
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH])
        y_ = tf.placeholder(tf.float32, [None, NUM_PER_IMAGE * LABEL_SIZE])

        # must be 4-D with shape `[batch_size, height, width, channels]`
        x_image = tf.reshape(x, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
        tf.summary.image('input', x_image, max_outputs=LABEL_SIZE)

    # define the model
    with tf.name_scope('convolution-layer-1'):
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])

        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

    with tf.name_scope('convolution-layer-2'):
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])

        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

    with tf.name_scope('densely-connected'):
        W_fc1 = weight_variable([IMAGE_WIDTH * IMAGE_HEIGHT * 4, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, IMAGE_WIDTH * IMAGE_HEIGHT * 4])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    with tf.name_scope('dropout'):
        # To reduce overfitting, we will apply dropout before the readout layer
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    with tf.name_scope('readout'):
        W_fc2 = weight_variable([1024, LABEL_SIZE])
        b_fc2 = bias_variable([LABEL_SIZE])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    # Define loss and optimizer
    # Returns:
    # A 1-D `Tensor` of length `batch_size`
    # of the same type as `logits` with the softmax cross entropy loss.


        # forword prop
    with tf.name_scope('forword-prop'):
        predict = tf.argmax(y_conv, axis=1)
        expect = tf.argmax(y_, axis=1)

        # evaluate accuracy
    with tf.name_scope('evaluate_accuracy'):
        correct_prediction = tf.equal(predict, expect)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        variable_summaries(accuracy)

    fixed_adv_sample_get_op = stepll_adversarial_images(x,y_conv)

    with tf.Session() as sess:

        restore(sess)

        # 初始化 tf.global_variables_initializer().run()

        # Test
        test_x, test_y = train_data.next_batch(1)
        _predict = predict.eval(feed_dict={x: test_x, y_: test_y, keep_prob: 1.0})
        _expect = expect.eval(feed_dict={x: test_x, y_: test_y, keep_prob: 1.0})
        _adv=fixed_adv_sample_get_op.eval(feed_dict={x: test_x, y_: test_y, keep_prob: 1.0})
        print(_adv.shape)
        _adv_predict = predict.eval(feed_dict={x: _adv, y_: test_y, keep_prob: 1.0})
        plt.subplot(1,2,1)
        plt.imshow(test_x[0])
        plt.subplot(1,2,2)
        plt.imshow(_adv[0])
        plt.show()
        print(_predict,_expect,_adv_predict,)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='images/char-4-epoch-1/',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
