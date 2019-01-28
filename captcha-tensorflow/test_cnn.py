# -*- coding:utf-8 -*-
import argparse
import datetime
import json
import numpy
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import datasets.base as input_data
import os
import tensorflow.contrib.slim as slim
os.environ["CUDA_VISIBLE_DEVICES"] = '1,2'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
MAX_STEPS = 10000
BATCH_SIZE = 1
TEST_BATCH_SIZE = 2000
LOG_DIR = 'log/cnn1-run-%s' % datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

FLAGS = None

meta = json.load(open('datasets/images/meta.json'))

LABEL_CHOICES= "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
print('data loaded')
LABEL_SIZE = 62
NUM_PER_IMAGE = 4
IMAGE_HEIGHT = 55
IMAGE_WIDTH = 110
IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT
print('label_size: %s, image_size: %s' % (LABEL_SIZE, IMAGE_SIZE))

x = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
y_ = tf.placeholder(tf.int32, [None, NUM_PER_IMAGE*LABEL_SIZE])
keep_prob = tf.placeholder(tf.float32)

def read_label(filename):
    basename = os.path.basename(filename)
    labels = basename.split('_')[0]

    data = []

    for c in labels:
        idx = LABEL_CHOICES.index(c)
        tmp = [0] * len(LABEL_CHOICES)
        tmp[idx] = 1
        data.extend(tmp)

    return data

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
            if f.endswith('.png'):
                imagepaths.append(os.path.join(path, f))
                labels.append(read_label(f))
                print((read_label(f)))

    print((labels))

    imagepaths = tf.convert_to_tensor(imagepaths, tf.string)
    labels = tf.convert_to_tensor(labels, tf.int32)

    # 建立 Queue
    imagepath, label = tf.train.slice_input_producer([imagepaths, labels], shuffle=shuffle)

    # 读取图片，并进行解码
    image = tf.read_file(imagepath)
    image = tf.image.decode_jpeg(image, channels=1)
    # 对图片进行裁剪和正则化（将数值[0,255]转化为[-1,1]）
    image = tf.multiply(tf.subtract(image / 255, 0.5), 2.0)
    image = tf.image.resize_images(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
    print(image.shape)
    image = tf.reshape(image, (IMAGE_HEIGHT, IMAGE_WIDTH,1))
    print(image.shape)

    # 创建 batch
    batch = tf.train.batch([image, label], batch_size=BATCH_SIZE, num_threads=1)

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


def CNN(input_data, keep_prob):
    end_point = {}
    resized = end_point['resized'] = tf.reshape(input_data, [-1, IMAGE_HEIGHT,IMAGE_WIDTH, 1])
    tf.summary.image('input', resized, max_outputs=LABEL_SIZE)

    conv1 = end_point['conv1'] = slim.conv2d(resized, 32, 3, padding='SAME', activation_fn=tf.nn.relu)
    pooling1 = end_point['pool1'] = slim.max_pool2d(conv1, 2)

    conv2 = end_point['conv2'] = slim.conv2d(pooling1, 64, 3, padding='SAME', activation_fn=tf.nn.relu)
    pooling2 = end_point['pool2'] = slim.max_pool2d(conv2, 2)

    flatten1 = end_point['flatten1'] = slim.flatten(pooling2)
    full1 = end_point['full1'] = slim.fully_connected(flatten1, 1024, activation_fn=tf.nn.relu)

    drop_out = end_point['drop_out'] = slim.dropout(full1, keep_prob)

    full2 = end_point['full2'] = slim.fully_connected(drop_out, NUM_PER_IMAGE*LABEL_SIZE, activation_fn=None)
    logits = end_point['logits'] = tf.reshape(full2, [-1, NUM_PER_IMAGE, LABEL_SIZE])
    predict = end_point['predict'] = tf.nn.softmax(logits)
    return end_point, logits, predict

y_expect_reshaped = tf.reshape(y_, [-1, NUM_PER_IMAGE, LABEL_SIZE])
end, log, pre = CNN(x, keep_prob)

with tf.name_scope('loss'):
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_expect_reshaped, logits=log))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    variable_summaries(cross_entropy)

# forword prop
with tf.name_scope('forword-prop'):
    predict = tf.argmax(log, axis=2)
    expect = tf.argmax(y_expect_reshaped, axis=2)

# evaluate accuracy

with tf.name_scope('evaluate_accuracy'):
    correct_prediction = tf.equal(predict, expect)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    variable_summaries(accuracy)



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

        plt.imshow(numpy.reshape(batch[0],(IMAGE_WIDTH,IMAGE_HEIGHT)), cmap='gray')
        plt.show()
        # print(batch[1])
        step_summary, _ = sess.run([merged, train_step], feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        train_writer.add_summary(step_summary, i)
        if i % 1000 == 0 and i!=0:
            # pass
            save(sess)
        if i % 100 == 0:
            # Test trained model
            batch = sess.run(test_data)
            test_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            print('testing accuracy = %.2f%%' % (test_accuracy * 100,))

            with open('model/log.txt', 'a') as f:
                f.write('testing accuracy = %.2f%%\n' % (test_accuracy * 100,))





