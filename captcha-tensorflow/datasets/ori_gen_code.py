import json
import string
import os
import shutil
import threading
import time
import uuid
from captcha.image import ImageCaptcha
import itertools

FLAGS = None
META_FILENAME = 'meta.json'
thread_len = 5
import random
import string
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import tensorflow as tf
font_path = 'Arial.ttf'
number = 4
# 80 25
size = (110, 55)
width, height = size
n_epoch = 1
num_per_image = 4

labels = list(string.ascii_letters)

for index in range(0, 10):
    labels.append(str(index))


def gene_text():
    return ''.join(random.sample(labels, number))

#
# def gene_code(code, fn):
#     width, height = size
#     image = Image.new('RGB', (width, height), color=(255, 255, 255))
#     font = ImageFont.truetype(font_path, size=25)
#     draw = ImageDraw.Draw(image)
#     font_width, font_height = font.getsize(code)
#     per_width = (width - font_width) / number + random.randint(-5, 20)
#     # per_width = (width - font_width) / number
#     per_height = (height - font_height) / number + random.randint(-5, 20)
#     draw.text(((per_width, per_height)), code,
#               font=font, fill=(0, 0, 0))
#     # image = image.filter(ImageFilter.EDGE_ENHANCE_MORE)
#     image = image.convert('L')
#     image.save(fn)
#     return image


# 为线程定义一个函数
class myThread(threading.Thread):
    def __init__(self, name, code_list, img_dir, width, height):
        threading.Thread.__init__(self)
        self.name = name
        self.code_list = code_list
        self.image = ImageCaptcha(width=width, height=height)
        self.img_dir = img_dir

    def run(self):
        print("开启线程：" + self.name)
        self.gen_code()
        print("退出线程：" + self.name)

    def gen_code(self):
        pass
        # todo

LABEL_CHOICES= "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

def read_label(code):
    data = []
    for c in code:
        idx = LABEL_CHOICES.index(c)
        tmp = [0] * len(LABEL_CHOICES)
        tmp[idx] = 1
        data.extend(tmp)
    return data
def _gen_captcha(img_dir, num_per_image, n=n_epoch):
    char_list = []
    threads = []
    # if os.path.exists(img_dir):
    #     shutil.rmtree(img_dir)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    s = time.time()
    print('generating %s epoches of captchas in %s' % (n_epoch, img_dir))
    # for _ in range(n):

    for num, i in enumerate(itertools.combinations(labels, num_per_image)):
        print(num)
        if num % 2 == 0:
            seg_dir = img_dir+'/batch_'+str(num)
            print(seg_dir)
            if not os.path.exists(seg_dir):
                os.mkdir(seg_dir)
            writer = tf.python_io.TFRecordWriter(seg_dir+"/train.tfrecords")
        captcha = ''.join(i)
        fn = os.path.join(seg_dir, '%s_%s.png' % (captcha, uuid.uuid4()))
        image=gene_code(captcha, fn)
        img_raw = image.tobytes()  # 将图片转化为原生bytes
        captcha=captcha.encode(encoding="utf-8")
        example = tf.train.Example(features=tf.train.Features(feature={
            "labels": tf.train.Feature(bytes_list=tf.train.BytesList(value=[captcha])),
            'images': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))
        writer.write(example.SerializeToString())  # 序列化为字符串
        if num==5:
            break
    writer.close()

    #         char_list.append(i)
    #
    # print(len(char_list))
    # per_thread=len(char_list)//thread_len
    # for i in range(thread_len):
    #     thread = myThread(name=str(i),img_dir=img_dir,code_list=char_list[i:i+per_thread],width=
    #     width,height=height,)
    #     thread.start()
    #     threads.append(thread)
    # thread = myThread(name=str(i+1), img_dir=img_dir, code_list=char_list[i+per_thread:], width=
    # width, height=height, )
    # thread.start()
    # threads.append(thread)x
    # for t in threads:
    #     t.join()
    print("退出主线程")
    e = time.time()
    print(e - s)

def build_file_path(x):
    if not os.path.isdir('images'):
        os.mkdir('images')
    return os.path.join('images', x)


def gen_dataset():
    # meta info
    meta = {
        'num_per_image': num_per_image,
        'label_size': len(labels),
        'label_choices': ''.join(labels),

        'n_epoch': n_epoch,
        'width': width,
        'height': height,
    }

    print('%s labels: %s' % (len(labels), ''.join(labels) or None))
    meta_filename = build_file_path(META_FILENAME)
    with open(meta_filename, 'w') as f:
        json.dump(meta, f, indent=4)
    print('write meta info in %s' % meta_filename)

    _gen_captcha(build_file_path('train'), num_per_image)
    _gen_captcha(build_file_path('test'), num_per_image, n=1)


if __name__ == '__main__':
    gen_dataset()
