## 参考博客
https://blog.csdn.net/u013733326/article/details/79639509

## 题目描述
```
搭建一个能够识别“猫”的简单的神经网络，利用logistic回归实现简单的二元分类
```

## 代码实现
### 库文件
```
* numpy ：是用Python进行科学计算的基本软件包。
* h5py：是与H5文件中存储的数据集进行交互的常用软件包。
* matplotlib：是一个著名的库，用于在Python中绘制图表。
* lr_utils ：自己简单编写的一个加载资料包里面的数据的简单功能的库。
```
#### lr_utils代码
Python:
```
import numpy as np
import h5py
    
def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes
```
- load_dataset()函数返回值
```
train_set_x_orig ：保存的是训练集里面的图像数据（本训练集有209张64x64的图像）。
train_set_y_orig ：保存的是训练集的图像对应的分类值（【0 | 1】，0表示不是猫，1表示是猫）。
test_set_x_orig ：保存的是测试集里面的图像数据（本训练集有50张64x64的图像）。
test_set_y_orig ： 保存的是测试集的图像对应的分类值（【0 | 1】，0表示不是猫，1表示是猫）。
classes ： 保存的是以bytes类型保存的两个字符串数据，数据为：[b’non-cat’ b’cat’]。
```

* 导入库
```
import numpy as np
import matplotlib.pyplot as plt
import h5py
from lr_utils import load_dataset
```