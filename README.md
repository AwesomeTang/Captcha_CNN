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
    width = 160  # 验证码图片的宽
    height = 60  # 验证码图片的高
    char_num = 4  # 验证码字符个数
    characters = range(10)	# 数字[0,9]
    test_folder = 'test'	# 测试集文件夹，下同
    train_folder = 'train'
    validation_folder = 'validation'
    tensorboard_folder = 'tensorboard'  # tensorboard的log路径
    generate_num = (5000, 500, 500)  # 训练集，验证集和测试集数量
    alpha = 1e-3  # 学习率
    Epoch = 100  # 训练轮次
    batch_size = 64     # 批次数量
    keep_prob = 0.5     # dropout比例
    print_per_batch = 20    # 每多少次输出结果
    save_per_batch = 20		# 每多少次写入tensorboard

```

##### 生成验证码（`class Generate`）

* 验证码图片示例：

![0478](/Users/tangwenpan/Documents/python/captcha_CNN/test/0478.jpg)

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

  ![image-20190113153215388](/Users/tangwenpan/Library/Application Support/typora-user-images/image-20190113153215388.png)

#### 训练&评估

* `next_batch()`：迭代器，分批次返还数据；
* `feed_data()`：给模型“喂”数据；
  * `x`：图像数组；
  * `y`：图像标签；
  * `keep_prob`：dropout比例；
* `evaluate()`：模型评估，用于验证集和测试集。
* `run_model()`：训练&评估

#### 目前效果

目前经过8000次迭代训练集准确率可达99%以上，测试集准确率89%，还是存在一定过拟合，不过现在模型是基于CPU训练的，完成一次训练耗费时间大约4个小时左右，后续调整了再进行更新。

```python
Images for train ：5000, for validation : 500, for test : 500
Epoch : 1
Step     0, train_acc:  11.72%, train_loss:  2.51, val_acc:  10.70%, val_loss:  2.82, improved:*  
Step    20, train_acc:  12.11%, train_loss:  0.33, val_acc:  10.00%, val_loss:  0.33, improved:   
Step    40, train_acc:  11.72%, train_loss:  0.50, val_acc:  10.30%, val_loss:  0.49, improved:   
Step    60, train_acc:  11.72%, train_loss:  0.33, val_acc:  10.60%, val_loss:  0.33, improved:   
```

* tensorboard

  * Accrracy

    ![image-20190113230650401](/Users/tangwenpan/Library/Application Support/typora-user-images/image-20190113230650401.png)

  * loss

    ![image-20190113230719042](/Users/tangwenpan/Library/Application Support/typora-user-images/image-20190113230719042.png)

后续待优化：

* 过拟合问题；
* 长时间陷入局部最优解。