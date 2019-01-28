import itertools
import os
import random
import time
import uuid
from constant import *
import tensorflow as tf
from PIL import Image, ImageFont, ImageDraw, ImageFilter
import matplotlib.pyplot as plt

def rndColor():
    return (random.randint(64,255),random.randint(64,255),random.randint(64,255))

def rndColor2():
    return (random.randint(32,127),random.randint(32,127),random.randint(32,127))

back_ground=rndColor()
width, height = (110,55)
font = ImageFont.truetype('Arial.ttf', size=25)
def gene_code_1(code, fn):
    font_width, font_height = font.getsize(code)
    image = Image.new('RGB', (width, height), color=(255,255,255))
    draw = ImageDraw.Draw(image)
    per_width = (width - font_width)/4
    per_height = (height - font_height) /4
    draw.text((per_width, per_height), code,
              font=font, fill=(0,0,0))

    return image

def gene_code_2(code, fn):
    print("level_2")

    font_width, font_height = font.getsize(code)
    image = Image.new('RGB', (width, height), color=back_ground)
    draw = ImageDraw.Draw(image)
    per_width = (width - font_width)/4 + random.randint(-5, 20)
    per_height = (height - font_height) / 4 + random.randint(-5, 20)
    for i in range(20):
        x1 = random.randint(0, width)
        y1 = random.randint(0, height)
        x2 = random.randint(0, width)
        y2 = random.randint(0, height)
        draw.line((x1, y1, x2, y2), fill=rndColor())
    draw.text((per_width, per_height), code,
              font=font, fill=rndColor2())

    return image
def gene_code_3(code, fn):
    print("level_3")
    font_width, font_height = font.getsize(code)
    image = Image.new('RGB', (width, height), color=back_ground)
    draw = ImageDraw.Draw(image)
    per_width = (width - font_width) / 4 + random.randint(-5, 20)
    per_height = (height - font_height) / 4 + random.randint(-5, 20)
    for i in range(20):
        x1 = random.randint(0, width)
        y1 = random.randint(0, height)
        x2 = random.randint(0, width)
        y2 = random.randint(0, height)
        draw.line((x1, y1, x2, y2), fill=rndColor())
    for i in range(80):
        draw.point([random.randint(0, width), random.randint(0, height)], fill=rndColor())
        x = random.randint(0, width)
        y = random.randint(0, height)
        draw.arc((x, y, x + 8, y + 8), 0, 90, fill=rndColor())
    # image = image.filter(ImageFilter.EDGE_ENHANCE_MORE)
    draw.text((per_width, per_height), code,
              font=font, fill=rndColor2())
    image = image.rotate(random.randint(-30, 30))
    draw = ImageDraw.Draw(image)
    for x in range(width):
        for y in range(height):
            c = image.getpixel((x, y))
            if c == (0, 0, 0):
                print(x, y)
                draw.point([x, y], fill=back_ground)
                print(image.getpixel((x, y)))
    # plt.imshow(image)
    # plt.show()
    # image.save(fn)
    return image
    # image = image.filter(ImageFilter.EDGE_ENHANCE_MORE)


def build_file_path(x):
    if not os.path.isdir('images'):
        os.mkdir('images')
    return os.path.join('images', x)
def _gen_captcha(img_dir, num_per_image):
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    s = time.time()
    print('generating %s epoches of captchas in %s' % (1, img_dir))
    for num, i in enumerate(itertools.combinations(LABEL_CHOICES, num_per_image)):
        print(num)
        if num % 50000 == 0:
            seg_dir = img_dir+'/batch_'+str(num)
            print(seg_dir)
            if not os.path.exists(seg_dir):
                os.mkdir(seg_dir)
            writer = tf.python_io.TFRecordWriter(seg_dir+"/train.tfrecords")
        captcha = ''.join(i)
        fn = os.path.join(seg_dir, '%s_%s.png' % (captcha,num))
        image=gene_code_2(captcha, fn)
        img_raw = image.tobytes()  # 将图片转化为原生bytes
        captcha=captcha.encode(encoding="utf-8")
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[captcha])),
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))
        writer.write(example.SerializeToString())  # 序列化为字符串
    writer.close()
    print("退出主线程")
    e = time.time()
    print(e - s)
_gen_captcha(build_file_path('train'), NUM_PER_IMAGE)
# _gen_captcha(build_file_path('test'), NUM_PER_IMAGE)