"""

"""

import datetime
import logging
import os
import time

import cv2
import numpy as np
import tensorflow as tf

import cnn_lstm_otc_ocr
import utils
import helper
from load_data import *

FLAGS = utils.FLAGS

logger = logging.getLogger('Traing for OCR using CNN+LSTM+CTC')
logger.setLevel(logging.INFO)


def train(train_dir=None, val_dir=None, mode='train'):
    model = cnn_lstm_otc_ocr.LSTMOCR(mode)
    model.build_graph()
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())


        num_batches_per_epoch = int(TRAIN_SET_NUM / FLAGS.batch_size)  # example: 100000/100
        num_batches_per_epoch_val = int(TRAIN_SET_NUM / FLAGS.batch_size)  # example: 10000/100

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
        train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)

        if FLAGS.restore:
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
            if ckpt:
                # the global_step will restore sa well
                saver.restore(sess, ckpt)
                print('restore from checkpoint{0}'.format(ckpt))

        print('=============================begin training=============================')
        for epoch in range(1000):
            train_feeder = sess.run(train_batch)
            batch_inputs,  batch_labels = \
                train_feeder[0],read_labels(train_feeder[1])
            feed = {model.inputs: batch_inputs,
                    model.labels: batch_labels}
            summary_str, batch_cost, step, _ = \
                sess.run([model.merged_summay, model.cost, model.global_step, model.train_op], feed)
            train_writer.add_summary(summary_str, step)
            if step % FLAGS.save_steps == 1:
                if not os.path.isdir(FLAGS.checkpoint_dir):
                    os.mkdir(FLAGS.checkpoint_dir)
                logger.info('save checkpoint at step {0}', format(step))
                saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'ocr-model'), global_step=step)
            # do validation
            if step % FLAGS.validation_steps == 0:
                acc_batch_total = 0
                lastbatch_err = 0
                lr = 0
                for j in range(num_batches_per_epoch_val):
                    val_inputs, val_labels = \
                        train_feeder[0], read_labels(train_feeder[1])

                    val_feed = {model.inputs: val_inputs,
                                model.labels: val_labels}

                    dense_decoded, lastbatch_err, lr = \
                        sess.run([model.dense_decoded, model.cost, model.lrn_rate],
                                 val_feed)
                    # print the decode result
                    print(dense_decoded)
                    print(val_labels)

                    acc = utils.accuracy_calculation(val_labels, dense_decoded,
                                                     ignore_value=-1, isPrint=True)
                    acc_batch_total += acc

                accuracy = (acc_batch_total * FLAGS.batch_size) / 2


                # train_err /= num_train_samples
                now = datetime.datetime.now()
                log = "{}/{} {}:{}:{} Epoch {}/{}, " \
                      "accuracy = {:.3f},avg_train_cost = {:.3f}, " \
                      "lastbatch_err = {:.3f}, time = {:.3f},lr={:.8f}"
                print(log.format(now.month, now.day, now.hour, now.minute, now.second,
                                 epoch + 1, FLAGS.num_epochs, accuracy, epoch,
                                 lastbatch_err, time.time() - epoch, lr))


def infer(img_path, mode='infer'):
    # imgList = load_img_path('/home/yang/Downloads/FILE/ml/imgs/image_contest_level_1_validate/')
    imgList = helper.load_img_path(img_path)
    print(imgList[:5])

    model = cnn_lstm_otc_ocr.LSTMOCR(mode)
    model.build_graph()

    total_steps = len(imgList) / FLAGS.batch_size

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
        ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        if ckpt:
            saver.restore(sess, ckpt)
            print('restore from ckpt{}'.format(ckpt))
        else:
            print('cannot restore')

        decoded_expression = []
        for curr_step in range(total_steps):

            imgs_input = []
            seq_len_input = []
            for img in imgList[curr_step * FLAGS.batch_size: (curr_step + 1) * FLAGS.batch_size]:
                im = cv2.imread(img, 0).astype(np.float32) / 255.
                im = np.reshape(im, [FLAGS.image_height, FLAGS.image_width, FLAGS.image_channel])

                def get_input_lens(seqs):
                    length = np.array([FLAGS.max_stepsize for _ in seqs], dtype=np.int64)

                    return seqs, length

                inp, seq_len = get_input_lens(np.array([im]))
                imgs_input.append(im)
                seq_len_input.append(seq_len)

            imgs_input = np.asarray(imgs_input)
            seq_len_input = np.asarray(seq_len_input)
            seq_len_input = np.reshape(seq_len_input, [-1])

            feed = {model.inputs: imgs_input}
            dense_decoded_code = sess.run(model.dense_decoded, feed)

            for item in dense_decoded_code:
                expression = ''

                for i in item:
                    if i == -1:
                        expression += ''
                    else:
                        expression += utils.decode_maps[i]

                decoded_expression.append(expression)

        with open('./result.txt', 'a') as f:
            for code in decoded_expression:
                f.write(code + '\n')


def main(_):
    if FLAGS.num_gpus == 0:
        dev = '/cpu:0'
    elif FLAGS.num_gpus == 1:
        dev = '/gpu:0'
    else:
        raise ValueError('Only support 0 or 1 gpu.')
    FLAGS.restore=True
    with tf.device(dev):
        if FLAGS.mode == 'train':
            train(FLAGS.train_dir, FLAGS.val_dir, FLAGS.mode)

        elif FLAGS.mode == 'infer':
            infer(FLAGS.infer_dir, FLAGS.mode)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
