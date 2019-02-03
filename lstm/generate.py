import os
import random
import time
from constant import *
from PIL import Image, ImageFont, ImageDraw, ImageFilter


def rndColor():
    return (random.randint(64, 255), random.randint(64, 255), random.randint(64, 255))


def rndColor2():
    return (random.randint(32, 127), random.randint(32, 127), random.randint(32, 127))


back_ground = rndColor()

font = ImageFont.truetype('Arial.ttf', size=25)


def gene_code_1(code, fn):
    font_width, font_height = font.getsize(code)
    image = Image.new('RGB', (IMAGE_WIDTH, IMAGE_HEIGHT), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    per_width = (IMAGE_WIDTH - font_width) / 4
    per_height = (IMAGE_HEIGHT - font_height) / 4
    draw.text((per_width, per_height), code,
              font=font, fill=(0, 0, 0))
    image.save(fn)


def gene_code_2(code, fn):
    print("level_2")
    font_width, font_height = font.getsize(code)
    image = Image.new('RGB', (IMAGE_WIDTH, IMAGE_HEIGHT), color=back_ground)
    draw = ImageDraw.Draw(image)
    per_width = (IMAGE_WIDTH - font_width) / 4 + random.randint(-5, 20)
    per_height = (IMAGE_HEIGHT - font_height) / 4 + random.randint(-5, 20)
    for i in range(20):
        x1 = random.randint(0, IMAGE_WIDTH)
        y1 = random.randint(0, IMAGE_HEIGHT)
        x2 = random.randint(0, IMAGE_WIDTH)
        y2 = random.randint(0, IMAGE_HEIGHT)
        draw.line((x1, y1, x2, y2), fill=rndColor())
    draw.text((per_width, per_height), code,
              font=font, fill=rndColor2())
    image.save(fn)


def gene_code_3(code, fn):
    print("level_3")
    font_width, font_height = font.getsize(code)
    image = Image.new('RGB', (IMAGE_WIDTH, IMAGE_HEIGHT), color=back_ground)
    draw = ImageDraw.Draw(image)
    per_width = (IMAGE_WIDTH - font_width) / 4 + random.randint(-5, 20)
    per_height = (IMAGE_HEIGHT - font_height) / 4 + random.randint(-5, 20)
    for i in range(20):
        x1 = random.randint(0, IMAGE_WIDTH)
        y1 = random.randint(0, IMAGE_HEIGHT)
        x2 = random.randint(0, IMAGE_WIDTH)
        y2 = random.randint(0, IMAGE_HEIGHT)
        draw.line((x1, y1, x2, y2), fill=rndColor())
    for i in range(80):
        draw.point([random.randint(0, IMAGE_WIDTH), random.randint(0, IMAGE_HEIGHT)], fill=rndColor())
        x = random.randint(0, IMAGE_WIDTH)
        y = random.randint(0, IMAGE_HEIGHT)
        draw.arc((x, y, x + 8, y + 8), 0, 90, fill=rndColor())
    # image = image.filter(ImageFilter.EDGE_ENHANCE_MORE)
    draw.text((per_width, per_height), code,
              font=font, fill=rndColor2())
    image = image.rotate(random.randint(-30, 30))
    draw = ImageDraw.Draw(image)
    for x in range(IMAGE_WIDTH):
        for y in range(IMAGE_HEIGHT):
            c = image.getpixel((x, y))
            if c == (0, 0, 0):
                draw.point([x, y], fill=back_ground)
    image.save(fn)
    # image = image.filter(ImageFilter.EDGE_ENHANCE_MORE)


def build_file_path(x):
    if not os.path.isdir('../imgs'):
        os.mkdir('../imgs')
    return os.path.join('../imgs', x)

def train_gen_captcha(train_set):
    img_dir = build_file_path('train')
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    s = time.time()
    print('generating %s epoches of captchas in %s' % (1, img_dir))
    # for num in range(train_set):
    #     print(num)
    #     slice = random.sample(LABEL_CHOICES_LIST, 4)
    #     captcha = ''.join(slice)
    #     fn = os.path.join(img_dir, '%s_%s.png' % (num, captcha))
    #     gene_code_1(captcha, fn)
    # for num in range(train_set):
    #     slice = random.sample(LABEL_CHOICES_LIST, 4)
    #     captcha = ''.join(slice)
    #     fn = os.path.join(img_dir, '%s_%s.png' % (num, captcha))
    #     gene_code_2(captcha, fn)
    for num in range(train_set):
        print(num)
        slice = random.sample(LABEL_CHOICES_LIST, 4)
        captcha = ''.join(slice)
        fn = os.path.join(img_dir, '%s_%s.png' % (num, captcha))
        gene_code_3(captcha, fn)
    print("退出主线程")
    e = time.time()
    print(e - s)

def val_gen_captcha(set_num):
    img_dir = build_file_path('val')
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    s = time.time()
    print('generating %s epoches of captchas in %s' % (1, img_dir))
    # for num in range(set_num):
    #     print(num)
    #     slice = random.sample(LABEL_CHOICES_LIST, 4)
    #     captcha = ''.join(slice)
    #     fn = os.path.join(img_dir, '%s_%s.png' % (num, captcha))
    #     gene_code_1(captcha, fn)
    # for num in range(set_num):
    #     slice = random.sample(LABEL_CHOICES_LIST, 4)
    #     captcha = ''.join(slice)
    #     fn = os.path.join(img_dir, '%s_%s.png' % (num, captcha))
    #     gene_code_2(captcha, fn)
    for num in range(set_num):
        slice = random.sample(LABEL_CHOICES_LIST, 4)
        captcha = ''.join(slice)
        fn = os.path.join(img_dir, '%s_%s.png' % (num, captcha))
        gene_code_3(captcha, fn)
    print("退出主线程")
    e = time.time()
    print(e - s)

train_gen_captcha(100000)
val_gen_captcha(10000)
