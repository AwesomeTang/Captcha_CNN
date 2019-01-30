# -*- coding: utf-8 -*-

# @author: Awesome_Tang
# @date: 2019-01-05
# @version: python2.7

from captcha.image import ImageCaptcha
import os
import random
from tqdm import tqdm
from PIL import Image
import numpy as np


class Config(object):
    width = 160  # 验证码图片的宽
    height = 60  # 验证码图片的高
    char_num = 4  # 验证码字符个数
    characters = range(10)
    generate_num = (10000, 500, 500)  # 训练集，验证集和测试集数量

    test_folder = 'test'
    train_folder = 'train'
    validation_folder = 'validation'
    tensorboard_folder = 'tensorboard'  # tensorboard的log路径
    saver_folder = 'checkpoints'

    alpha = 1e-3  # 学习率
    Epoch = 100  # 训练轮次
    batch_size = 64  # 批次数量
    keep_prob = 0.5  # dropout比例
    print_per_batch = 20  # 每多少次输出结果
    save_per_batch = 20


class Generate:

    def __init__(self):
        self.image = ImageCaptcha(width=Config.width, height=Config.height)
        self.check_path(Config.test_folder)
        self.check_path(Config.validation_folder)
        self.check_path(Config.train_folder)
        self.run()

    @staticmethod
    def check_path(folder):
        # 检查文件夹是否存在，不存在就创建
        if os.path.exists(folder):
            pass
        else:
            os.mkdir(folder)

    def gen_captcha(self, folder, gen_num, random_=True):
        # 生成验证码图片
        desc = '{:<10}'.format(folder)

        if random_:
            # 随机生成验证码，用于测试集和验证集
            for _ in tqdm(range(gen_num), desc=desc):
                while True:
                    label = ''.join('%s' % num for num in
                                    random.sample(Config.characters, Config.char_num))
                    path = folder + '/%s.jpg' % label

                    # 检查验证码是否已存在
                    if not os.path.exists(path):
                        self.image.generate_image(label)
                        self.image.write(label, path)
                        break

        else:
            # 按顺序生成验证码
            for num in tqdm(range(gen_num), desc=desc):
                num_length = len(str(num))

                if num_length < Config.char_num:
                    # 不足4位由0补齐
                    label = '0' * (Config.char_num - num_length) + str(num)
                    path = folder + '/%s.jpg' % label
                    self.image.generate_image(label)
                    self.image.write(label, path)
                else:
                    label = str(num)
                    path = folder + '/%s.jpg' % label
                    self.image.generate_image(label)
                    self.image.write(label, path)

    def run(self):
        print '==> Generating images...'
        self.gen_captcha(Config.train_folder, Config.generate_num[0], random_=False)
        self.gen_captcha(Config.validation_folder, Config.generate_num[1])
        self.gen_captcha(Config.test_folder, Config.generate_num[2])


class ReadData:

    def __init__(self):
        self.test_img = os.listdir(Config.test_folder)
        self.train_img = os.listdir(Config.train_folder)
        self.sample_num = len(self.train_img)

    def read_data(self, path):
        img = Image.open(path).convert('L')
        image_array = np.array(img)
        image_data = image_array.flatten() / 255.0
        # 切割图片路径
        label = os.path.splitext(os.path.split(path)[1])[0]
        label_vec = self.label2vec(label)
        return image_data, label_vec

    @staticmethod
    def label2vec(label):
        """
        将验证码标签转为40维的向量。
        :param label: 1327
        :return:
            [0,1,0,0,0,0,0,0,0,0,
            0,0,0,1,0,0,0,0,0,0,
            0,0,1,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,1,0,0]
        """
        label_vec = np.zeros(Config.char_num * len(Config.characters))
        for i, num in enumerate(label):
            idx = i * len(Config.characters) + int(num)
            label_vec[idx] = 1
        return label_vec

    def load_data(self, folder):
        """
        加载样本数据
        :param folder: 图片存放文件夹
        :return:
            data: 图片数据
            label:  图片标签
            size:   图片数量
        """
        if os.path.exists(folder):
            path_list = os.listdir(folder)
            size = len(path_list)
            data = np.zeros([size, Config.height * Config.width])
            label = np.zeros([size, Config.char_num * len(Config.characters)])
            for i, img_path in enumerate(path_list):
                path = '%s/%s' % (folder, img_path)
                data[i], label[i] = self.read_data(path)
            return data, label, size
        else:
            raise IOError('No such directory, please check again.')


if __name__ == '__main__':
    Generate()
