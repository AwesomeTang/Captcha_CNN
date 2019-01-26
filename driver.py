# -*- coding: utf-8 -*-

# @author: Awesome_Tang
# @date: 2019-01-05
# @version: python2.7


from cnn_model import *
from Generate_Captcha import *
import os
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Run:

    def __init__(self):
        read = ReadData()
        self.test_x, self.test_y, self.test_num = read.load_data(folder=Config.test_folder)
        self.train_x, self.train_y, self.train_num = read.load_data(folder=Config.train_folder)
        self.val_x, self.val_y, self.val_num = read.load_data(folder=Config.validation_folder)

        print 'Images for train ：%d, for validation : %d, for test : %d' \
              % (self.train_num, self.val_num, self.test_num)

        self.run_model()

    @staticmethod
    def next_batch(x, y, length):
        if length % Config.batch_size == 0:
            times = int(length / Config.batch_size)
        else:
            times = int(length / Config.batch_size) + 1

        start_id = 0
        for _ in range(times):
            end_id = min(start_id + Config.batch_size, length)
            batch_data = x[start_id:end_id]
            batch_label = y[start_id:end_id]
            start_id = end_id
            yield batch_data, batch_label

    @staticmethod
    def feed_data(x, y, keep_prob, is_training=True):
        dict = {model.input_x: x,
                model.input_y: y,
                model.keep_prob: keep_prob,
                model.training: is_training}
        return dict

    def evaluate(self, sess, val_x, val_y, val_size):
        total_loss = 0.
        total_acc = 0.

        for x_, y_ in self.next_batch(val_x, val_y, val_size):
            length = len(y_)
            dict = self.feed_data(x_, y_, 1.0, False)
            val_acc, val_loss = sess.run([model.accuracy, model.loss], feed_dict=dict)
            total_acc += val_acc * length
            total_loss += val_loss * length
        return total_acc / val_size, total_loss / val_size

    def run_model(self):

        saver = tf.train.Saver(max_to_keep=1)
        if not os.path.exists(Config.saver_folder):
            os.mkdir(Config.saver_folder)
        save_path = os.path.join(Config.saver_folder, 'best_validation')

        total_batch = 0
        best_acc = 0
        last_improved_step = 0
        require_steps = 1000
        flag = False
        start_time = datetime.now()

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        for epoch in range(Config.Epoch):
            print 'Epoch : %d' % (epoch + 1)
            for x, y in self.next_batch(self.train_x, self.train_y, self.train_num):
                dict = self.feed_data(x, y, Config.keep_prob, True)
                sess.run(model.train_step, feed_dict=dict)

                if total_batch % Config.print_per_batch == 0:
                    # 输出在验证集和训练集上的准确率和损失值
                    dict[model.keep_prob] = 1.0
                    dict[model.training] = False
                    train_accuracy, train_loss = sess.run([model.accuracy, model.loss],
                                                          feed_dict=dict)
                    val_acc, val_loss = self.evaluate(sess, self.val_x, self.val_y, self.val_num)

                    if val_acc > best_acc:
                        # 记录最好的结果
                        best_acc = val_acc
                        last_improved_step = total_batch
                        # 保存模型
                        saver.save(sess=sess, save_path=save_path)
                        improved = '*'
                    else:
                        improved = ''

                    msg = 'Step {:5}, train_acc:{:8.2%}, train_loss:{:6.2f},' \
                          ' val_acc:{:8.2%}, val_loss:{:6.2f}, improved:{:3}'
                    print msg.format(total_batch, train_accuracy, train_loss, val_acc, val_loss, improved)

                if total_batch % Config.save_per_batch == 0:
                    # 写入tensorboard
                    dict[model.keep_prob] = 1.0
                    dict[model.training] = False
                    s = sess.run(model.merged_summary, feed_dict=dict)
                    model.writer.add_summary(s, total_batch)

                if total_batch - last_improved_step > require_steps:
                    flag = True
                    break

                total_batch += 1
            if flag:
                print 'No improvement for over %d steps, auto-stopping....'%require_steps
                break
        end_time = datetime.now()
        time_diff = (end_time - start_time).seconds
        print 'Time Usage : {:.2f} hours'.format(time_diff / 3600.0)
        # 输出在测试集上的准确率
        test_acc, test_loss = self.evaluate(sess, self.test_x, self.test_y, self.test_num)

        print "Test accuracy:{:8.2%}, loss:{:6.2f}".format(test_acc, test_loss)
        sess.close()


if __name__ == "__main__":
    model = CNN()
    Run()
