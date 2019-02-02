import os
import numpy as np
import tensorflow as tf
from constant  import *
cwd=os.getcwd()

def read_labels(codes):
    data = []
    for row in codes:
        code_row = []
        row = row.decode('utf8')
        for c in row:
            idx = LABEL_CHOICES.index(c)
            tmp = [0] * len(LABEL_CHOICES)
            tmp[idx] = 1
            code_row.extend(tmp)
        data.append(code_row)
    return data

def test_read_and_decode():
    files_name= [cwd+'/images/test/'+i for i in os.listdir(cwd+'/images/test')]

    filename_queue = tf.train.string_input_producer(files_name)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={"label": tf.FixedLenFeature([], tf.string),
                                                 "image": tf.FixedLenFeature([], tf.string)})
    label = features["label"]
    img = tf.decode_raw(features['image'], tf.uint8)
    # img = tf.image.resize_images(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
    img = tf.cast(img,tf.float32)
    img = tf.reshape(img,[IMAGE_HEIGHT, IMAGE_WIDTH, 3])
    preprocessed = tf.multiply(tf.subtract(img / 255, 0.5), 2.0)
    batch = tf.train.batch([preprocessed, label], batch_size=BATCH_SIZE,
        capacity=10000,  num_threads=64)
    return batch

def train_read_and_decode():
    # files_name= [cwd+'/images/train/'+i for i in os.listdir(cwd+'/images/train') if i.endswith('tfrecords')]
    files_name= ['D:/pyproject/ewriopvkj/captcha-tensorflow/datasets/images/train/train.tfrecords']

    filename_queue = tf.train.string_input_producer(files_name)
    print(files_name)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={"label": tf.FixedLenFeature([], tf.string),
                                                 "image": tf.FixedLenFeature([], tf.string)})
    label = features["label"]
    img = tf.decode_raw(features['image'], tf.uint8)

    img = tf.reshape(img,[IMAGE_HEIGHT, IMAGE_WIDTH, 3])
    img = tf.cast(img,tf.float32)
    img = tf.multiply(tf.subtract(img / 255, 0.5), 2.0)
    batch = tf.train.shuffle_batch([img, label], batch_size=BATCH_SIZE,
        capacity=10000, min_after_dequeue=5000, num_threads=64)
    return batch

train_batch=train_read_and_decode()

sess = tf.Session()

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())


import matplotlib.pyplot as plt

batch=sess.run(train_batch)
print(batch[1][0])
# plt.imshow(batch[0][0])
plt.show()