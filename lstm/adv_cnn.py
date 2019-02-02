import os
import cv2
import numpy as np
import tensorflow as tf
import cnn_lstm_otc_ocr
import utils
import helper
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

FLAGS = utils.FLAGS
FLAGS.batch_size = 1

imgs_input = []
#图片所在路径
imgList = helper.load_img_path('./infer/')
print(imgList[:5])
for img in imgList:
    im = cv2.imread(img, cv2.IMREAD_COLOR).astype(np.float32) / 255.
    im = np.reshape(im, [FLAGS.image_height, FLAGS.image_width, FLAGS.image_channel])
    imgs_input.append(im)
imgs_input = np.asarray(imgs_input)
model = cnn_lstm_otc_ocr.LSTMOCR('infer')
model.build_graph()

#定义节点
logit = model.get_logist()

with tf.Session(config=config) as sess:

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
    ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)

    saver.restore(sess, ckpt)
    print('restore from ckpt  {}'.format(ckpt))

    feed = {model.inputs: imgs_input}
    dense_decoded_code = sess.run(model.dense_decoded, feed)

    decoded_expression= utils.read_labels(dense_decoded_code)

    print(sess.run(logit,feed))
for code in decoded_expression:
    print('result:'+code)

