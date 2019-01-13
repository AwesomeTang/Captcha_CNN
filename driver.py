# -*- coding: utf-8 -*-

# @author: Awesome_Tang
# @date: 2019-01-05
# @version: python2.7


from cnn_model import *
from Generate_Captcha import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Run:

    def __init__(self):
        read = ReadData()
        self.test_x, self.test_y, self.test_num = read.load_data(folder=Config.test_folder)
        self.train_x, self.train_y, self.train_num = read.load_data(folder=Config.train_folder)
        self.val_x, self.val_y, self.val_num = read.load_data(folder=Config.validation_folder)

        print 'Images for train ：%d, for validation : %d, for test : %d'\
              % (self.train_num, self.val_num, self.test_num)

        self.run_model()

    def next_batch(self, x, y, length):
        if length % Config.batch_size == 0:
            times = int(length / Config.batch_size)
        else:
            times = int(length / Config.batch_size) + 1

        start_id = 0
        for _ in range(times):
            end_id = min(start_id + Config.batch_size,length)
            batch_data = x[start_id:end_id]
            batch_label = y[start_id:end_id]
            start_id = end_id
            yield batch_data, batch_label

    def feed_data(self, x, y, keep_prob):
        dict = {model.input_x: x,
                model.input_y: y,
                model.keep_prob: keep_prob}
        return dict

    def evaluate(self, sess, val_x, val_y, val_size):
        total_loss = 0.
        total_acc = 0.

        for x_, y_ in self.next_batch(val_x, val_y, val_size):
            length = len(y_)
            dict = self.feed_data(x_, y_, 1.0)
            val_acc, val_loss = sess.run([model.accuracy, model.loss], feed_dict=dict)
            total_acc += val_acc * length
            total_loss += val_loss * length
        return total_acc / val_size, total_loss / val_size

    def run_model(self):

        total_batch = 0
        best_acc = 0
        last_improved_step = 0
        flag = False

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        for epoch in range(Config.Epoch):
            print 'Epoch : %d' % (epoch + 1)
            for x, y in self.next_batch(self.train_x, self.train_y, self.train_num):
                dict = self.feed_data(x, y, Config.keep_prob)
                sess.run(model.train_step, feed_dict=dict)

                if total_batch % Config.print_per_batch == 0:
                    dict[model.keep_prob] = 1.0
                    train_accuracy, train_loss = sess.run([model.accuracy, model.loss], feed_dict=dict)
                    val_acc, val_loss = self.evaluate(sess, self.val_x, self.val_y, self.val_num)

                    if val_acc > best_acc:
                        best_acc = val_acc
                        last_improved_step = total_batch
                        improved = '*'
                    else:
                        improved = ''

                    msg = 'Step {:5}, train_acc:{:8.2%}, train_loss:{:6.2f},' \
                          ' val_acc:{:8.2%}, val_loss:{:6.2f}, improved:{:3}'
                    print msg.format(total_batch, train_accuracy, train_loss, val_acc, val_loss, improved)

                if total_batch % Config.save_per_batch == 0:
                    dict[model.keep_prob] = 1.0
                    s = sess.run(model.merged_summary, feed_dict=dict)
                    model.writer.add_summary(s, total_batch)

                if total_batch - last_improved_step > 1000:
                    flag = True
                    break

                total_batch += 1
            if flag:
                print 'No improvement for over 1000 steps, auto-stopping....'
                break

        test_acc, test_loss = self.evaluate(sess, self.test_x, self.test_y, self.test_num)

        print "Test accuracy:{:8.2%}, accuracy:{:6.2f}".format(test_acc, test_loss)
        sess.close()


if __name__ == "__main__":
    model = CNN()
    Run()
