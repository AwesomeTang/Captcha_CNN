完整代码：[GitHub](https://github.com/AwesomeTang/Captcha_CNN)

我的简书：[Awesome_Tang的简书](https://www.jianshu.com/u/27c6d3dbb54f)

整个项目代码分为三部分：

* `Generrate_Captcha`:
  * 生成验证码图片（训练集，验证集和测试集）；
  * 读取图片数据和标签（标签即为图片文件名）；
* `cnn_model`:卷积神经网络；
* `driver`:模型训练及评估。

#### Generate Captcha

##### 配置项

```python
class Config(object):
    width = 160 # 验证码图片的宽
    height = 60 # 验证码图片的高
    char_num = 4    # 验证码字符个数
    characters = range(10)
    test_folder = 'test'
    train_folder = 'train'
    validation_folder = 'validation'
    tensorboard_folder = 'tensorboard'  # tensorboard的log路径
    generate_num = (10000, 500, 500)    # 训练集，验证集和测试集数量
    alpha = 1e-3    # 学习率
    Epoch = 100 # 训练轮次
    batch_size = 64 # 批次数量
    keep_prob = 0.5 # dropout比例
    print_per_batch = 20    # 每多少次输出结果
    save_per_batch = 20	# 每多少次写入tensoboard

```

##### 生成验证码（`class Generate`）

* 验证码图片示例：

![0193](https://github.com/AwesomeTang/Captcha_CNN/blob/master/images/0193.jpg)

* `check_path()`:检查文件夹是否存在，如不存在则创建。
* `gen_captcha()`:生成验证码方法，写入之前检查是否以存在，如存在重新生成。

#### 读取数据（`classs ReadData`）

* `read_data()`:返回图片数组（`numpy.array`格式）和标签（即文件名）；

* `label2vec()`:将文件名转为向量；

  * 例：

    ```python
    label = '1327'
    
    label_vec = [0,1,0,0,0,0,0,0,0,0,
    		    0,0,0,1,0,0,0,0,0,0,
    		    0,0,1,0,0,0,0,0,0,0,
    		    0,0,0,0,0,0,0,1,0,0]
    ```

* `load_data()`:加载文件夹下所有图片，返回图片数组，标签和图片数量。

#### 定义模型（`cnn_model`）

采用三层卷积，`filter_size`均为5，为避免过拟合，每层卷积后面均接`dropout`操作，最终将$160*60$的图像转为$20*8$的矩阵。

* 大致结构如下：

  ![整体结构](https://github.com/AwesomeTang/Captcha_CNN/blob/master/images/image-20190113153215388.png)

#### 训练&评估

* `next_batch()`：迭代器，分批次返还数据；
* `feed_data()`：给模型“喂”数据；
  * `x`：图像数组；
  * `y`：图像标签；
  * `keep_prob`：dropout比例；
* `evaluate()`：模型评估，用于验证集和测试集。
* `run_model()`：训练&评估

#### 目前效果

目前经过8000次迭代训练集准确率可达99%以上，测试集准确率93%，还是存在一定过拟合，不过现在模型是基于CPU训练的，完成一次训练耗费时间大约4个小时左右，后续调整了再进行更新。

```
Images for train ：10000, for validation : 1000, for test : 1000
Epoch : 1
Step     0, train_acc:   7.42%, train_loss:  1.43, val_acc:   9.85%, val_loss:  1.40, improved:*  
Step    20, train_acc:  12.50%, train_loss:  0.46, val_acc:  10.35%, val_loss:  0.46, improved:*  
Step    40, train_acc:   9.38%, train_loss:  0.37, val_acc:  10.10%, val_loss:  0.37, improved:   
Step    60, train_acc:   7.42%, train_loss:  0.34, val_acc:  10.25%, val_loss:  0.34, improved:   
Step    80, train_acc:   7.81%, train_loss:  0.33, val_acc:   9.82%, val_loss:  0.33, improved:   
Step   100, train_acc:  12.11%, train_loss:  0.33, val_acc:  10.00%, val_loss:  0.33, improved:   
Step   120, train_acc:   9.77%, train_loss:  0.33, val_acc:  10.07%, val_loss:  0.33, improved:   
Step   140, train_acc:   8.98%, train_loss:  0.33, val_acc:  10.40%, val_loss:  0.33, improved:*  
Epoch : 2
Step   160, train_acc:   8.20%, train_loss:  0.33, val_acc:  10.52%, val_loss:  0.33, improved:*  
...
Epoch : 51
Step  7860, train_acc: 100.00%, train_loss:  0.01, val_acc:  92.37%, val_loss:  0.08, improved:   
Step  7880, train_acc:  99.61%, train_loss:  0.01, val_acc:  92.28%, val_loss:  0.08, improved:   
Step  7900, train_acc: 100.00%, train_loss:  0.01, val_acc:  92.42%, val_loss:  0.08, improved:   
Step  7920, train_acc: 100.00%, train_loss:  0.00, val_acc:  92.83%, val_loss:  0.08, improved:   
Step  7940, train_acc: 100.00%, train_loss:  0.01, val_acc:  92.77%, val_loss:  0.08, improved:   
Step  7960, train_acc: 100.00%, train_loss:  0.01, val_acc:  92.68%, val_loss:  0.08, improved:   
Step  7980, train_acc: 100.00%, train_loss:  0.00, val_acc:  92.63%, val_loss:  0.09, improved:   
No improvement for over 1000 steps, auto-stopping....
Test accuracy:  93.00%, loss:  0.08
           
```

* tensorboard

  * Accrracy

    ![accrracy](https://github.com/AwesomeTang/Captcha_CNN/blob/master/images/acc.png)

  * loss

    ![loss](https://github.com/AwesomeTang/Captcha_CNN/blob/master/images/loss.png)

