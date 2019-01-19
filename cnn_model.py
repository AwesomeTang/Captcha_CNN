# -*- coding: utf-8 -*-

# @author: Awesome_Tang
# @date: 2019-01-05
# @version: python2.7

from Generate_Captcha import *
import tensorflow as tf


class CNN:
    def __init__(self):
        self.input_x = tf.placeholder(
            tf.float32, [None, Config.width * Config.height], name='input_x')
        self.input_y = tf.placeholder(
            tf.float32, [None, Config.char_num * len(Config.characters)], name='input_y')
        self.keep_prob = tf.placeholder("float")
        self.training = tf.placeholder(tf.bool)

        self.CNN_model()

    @staticmethod
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    @staticmethod
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    @staticmethod
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")

    @staticmethod
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


    def CNN_model(self):
        x_image = tf.reshape(self.input_x,
                             [-1, Config.height, Config.width, 1], name='x_image')
        # batch normalization
        x_norm = tf.layers.batch_normalization(x_image,
                                               training=self.training ,momentum=0.9)

        # 卷积层 1:
        w_cv1 = self.weight_variable([5, 5, 1, 32])
        b_cv1 = self.bias_variable([32])
        h_cv1 = tf.nn.relu(self.conv2d(x_norm, w_cv1) + b_cv1)
        h_mp1 = self.max_pool_2x2(h_cv1)
        #h_mp1 = tf.nn.dropout(h_mp1,Config.keep_prob)

        # 卷积层 2
        w_cv2 = self.weight_variable([5, 5, 32, 64])
        b_cv2 = self.bias_variable([64])
        h_cv2 = tf.nn.relu(self.conv2d(h_mp1, w_cv2) + b_cv2)
        h_mp2 = self.max_pool_2x2(h_cv2)
        #h_mp2 = tf.nn.dropout(h_mp2, Config.keep_prob)

        # 卷积层 3
        w_cv3 = self.weight_variable([5, 5, 64, 64])
        b_cv3 = self.bias_variable([64])
        h_cv3 = tf.nn.relu(self.conv2d(h_mp2, w_cv3) + b_cv3)
        h_mp3 = self.max_pool_2x2(h_cv3)
        # h_mp3 = tf.nn.dropout(h_mp3, Config.keep_prob)

        # 全连接
        W_fc1 = self.weight_variable([20 * 8 * 64, 128])
        b_fc1 = self.bias_variable([128])
        h_mp3_flat = tf.reshape(h_mp3, [-1, 20 * 8 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_mp3_flat, W_fc1) + b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        # 输出层
        W_fc2 = self.weight_variable([128, Config.char_num * len(Config.characters)])
        b_fc2 = self.bias_variable([Config.char_num * len(Config.characters)])
        output = tf.add(tf.matmul(h_fc1_drop, W_fc2), b_fc2)

        self.loss = (tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y, logits=output)))
        predict = tf.reshape(output, [-1, Config.char_num,
                                      len(Config.characters)], name='predict')
        labels = tf.reshape(self.input_y, [-1, Config.char_num,
                                           len(Config.characters)], name='labels')
        predict_max_idx = tf.argmax(predict, axis=2, name='predict_max_idx')
        labels_max_idx = tf.argmax(labels, axis=2, name='labels_max_idx')
        predict_correct_vec = tf.equal(predict_max_idx, labels_max_idx)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_step = tf.train.AdamOptimizer(
                Config.alpha).minimize(self.loss)
        self.accuracy = tf.reduce_mean(tf.cast(predict_correct_vec, tf.float32))

        # tensorboard 配置
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("accuracy", self.accuracy)
        self.merged_summary = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(Config.tensorboard_folder)
