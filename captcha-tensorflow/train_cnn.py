# -*- coding:utf-8 -*-
import argparse
import datetime
import json
import numpy
import sys
import tensorflow as tf

import datasets.base as input_data
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '2,3'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
MAX_STEPS = 10000
BATCH_SIZE = 1
TEST_BATCH_SIZE = 2000
LOG_DIR = 'log/cnn1-run-%s' % datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

FLAGS = None

meta = json.load(open('datasets/images/meta.json'))

print('data loaded')
LABEL_SIZE = meta['label_size']
NUM_PER_IMAGE = meta['num_per_image']
IMAGE_HEIGHT = meta['height']
IMAGE_WIDTH = meta['width']
IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT
print('label_size: %s, image_size: %s' % (LABEL_SIZE, IMAGE_SIZE))


def read_image(type='train'):
    if type == 'train':
        path = 'datasets/images/train'
        shuffle = True
    else:
        path = 'datasets/images/test'
        shuffle = False
    imagepaths = []
    labels = []

    for i, j, sample in os.walk(path):
        for f in sample:
            imagepaths.append(os.path.join(path, f))
            labels.append(f[:4])

    imagepaths = tf.convert_to_tensor(imagepaths, tf.string)
    labels = tf.convert_to_tensor(labels, tf.string)

    # 建立 Queue
    imagepath, label = tf.train.slice_input_producer([imagepaths, labels], shuffle=shuffle)

    # 读取图片，并进行解码
    image = tf.read_file(imagepath)
    image = tf.image.decode_jpeg(image, channels=1)
    # 对图片进行裁剪和正则化（将数值[0,255]转化为[-1,1]）
    image = tf.multiply(tf.subtract(image / 255, 0.5), 2.0)
    image = tf.image.resize_images(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
    image = tf.reshape(image, (IMAGE_WIDTH, IMAGE_HEIGHT))

    # 创建 batch
    batch = tf.train.batch([image, label], batch_size=BATCH_SIZE, num_threads=64)

    return batch


def save(sess, ):
    saver = tf.train.Saver()
    saver.save(sess, "model/" + FLAGS.data_dir + "/model.ckpt")


def restore(sess):
    saver = tf.train.Saver()
    saver.restore(sess, "model/" + FLAGS.data_dir + "/model.ckpt")


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


def net():
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, IMAGE_WIDTH, IMAGE_HEIGHT])
        y_ = tf.placeholder(tf.string, [None, 4])

        # must be 4-D with shape `[batch_size, height, width, channels]`
        x_image = tf.reshape(x, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
        tf.summary.image('input', x_image, max_outputs=LABEL_SIZE)
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
        W_fc2 = weight_variable([1024, NUM_PER_IMAGE * LABEL_SIZE])
        b_fc2 = bias_variable([NUM_PER_IMAGE * LABEL_SIZE])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    with tf.name_scope('reshape'):
        y_expect_reshaped = tf.reshape(y_, [-1, NUM_PER_IMAGE, LABEL_SIZE])
        y_got_reshaped = tf.reshape(y_conv, [-1, NUM_PER_IMAGE, LABEL_SIZE])

    # Define loss and optimizer
    # Returns:
    # A 1-D `Tensor` of length `batch_size`
    # of the same type as `logits` with the softmax cross entropy loss.
    with tf.name_scope('loss'):
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_expect_reshaped, logits=y_got_reshaped))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        variable_summaries(cross_entropy)

    # forword prop
    with tf.name_scope('forword-prop'):
        predict = tf.argmax(y_got_reshaped, axis=2)
        expect = tf.argmax(y_expect_reshaped, axis=2)

    # evaluate accuracy

    with tf.name_scope('evaluate_accuracy'):
        correct_prediction = tf.equal(predict, expect)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        variable_summaries(accuracy)


net()

train_data = read_image()
test_data = read_image('test')
t = tf.constant(100)
with tf.Session(config=config) as sess:
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(LOG_DIR + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(LOG_DIR + '/test', sess.graph)
    tf.global_variables_initializer().run()
    # 开启一个协调器
    coord = tf.train.Coordinator()
    # 启动队列填充才可以是用batch
    threads = tf.train.start_queue_runners(sess, coord)
    for i in range(MAX_STEPS):
        print(sess.run(t))
        batch = sess.run(train_data)
        import matplotlib.pyplot as plt

        # plt.imshow(numpy.reshape(batch[0],(110,55)), cmap='gray')
        # plt.show()
        step_summary, _ = sess.run([merged, train_step], feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        train_writer.add_summary(step_summary, i)
        if i % 1000 == 0:
            save(sess)
        if i % 100 == 0:
            # Test trained model
            valid_summary, train_accuracy = sess.run([merged, accuracy],
                                                     feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            print('step %s, training accuracy = %.2f%%' % (
                i, train_accuracy * 100))
    save(sess)
    train_writer.close()
    test_writer.close()

    # final check after looping
    batch = sess.run(train_data)
    test_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
    print('testing accuracy = %.2f%%' % (test_accuracy * 100,))
