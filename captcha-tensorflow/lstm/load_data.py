import os
import numpy as np
import tensorflow as tf
from constant  import *
charset = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
encode_maps = {}
decode_maps = {}
for i, char in enumerate(charset, 1):
    encode_maps[char] = i
    decode_maps[i] = char

SPACE_INDEX = 0
SPACE_TOKEN = ''
encode_maps[SPACE_TOKEN] = SPACE_INDEX
decode_maps[SPACE_INDEX] = SPACE_TOKEN
cwd=os.getcwd()+'/datasets/'

def sparse_tuple_from_label(sequences, dtype=np.int32):
    """Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape
def read_labels(codes):
    data = []
    for row in codes:
        code = row.decode('utf8')
        code = [SPACE_INDEX if code == SPACE_TOKEN else encode_maps[c] for c in list(code)]
        data.append(code)

    return sparse_tuple_from_label(data)

def test_read_and_decode():
    files_name= [cwd+'/test/'+i for i in os.listdir(cwd+'/test')if i.endswith('tfrecords')]

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
    files_name= [cwd+'/train/'+i for i in os.listdir(cwd+'/train') if i.endswith('tfrecords')]
    # files_name= ['D:/pyproject/ewriopvkj/captcha-tensorflow/datasets/images/train/train.tfrecords']

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
#
train_batch=train_read_and_decode()
test_batch=test_read_and_decode()

# sess = tf.Session()
#
# coord = tf.train.Coordinator()
# threads = tf.train.start_queue_runners(sess=sess, coord=coord)
# sess.run(tf.global_variables_initializer())
# sess.run(tf.local_variables_initializer())
#
#
# import matplotlib.pyplot as plt
#
# batch=sess.run(train_batch)
# print(batch[1][0])
# # plt.imshow(batch[0][0])
# plt.show()
python3 ./main.py --train_dir=../imgs/train/ \
  --val_dir=../imgs/val/ \
  --image_height=60 \
  --image_width=180 \
  --image_channel=3 \
  --out_channels=64 \
  --num_hidden=128 \
  --batch_size=128 \
  --log_dir=./log/train \
  --num_gpus=1 \
  --mode=train