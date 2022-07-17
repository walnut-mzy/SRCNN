import os
import tensorflow as tf
import numpy as np
import cv2
import settings
from sklearn.model_selection import train_test_split
from tqdm import *
def pic_fuzzy(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.IMREAD_COLOR)

    # 此为均值模糊
    # （30,1）为一维卷积核，指在x，y方向偏移多少位
    dst1 = cv2.blur(image, (30, 1))

    # 此为中值模糊，常用于去除椒盐噪声
    dst2 = cv2.medianBlur(image, 15)

    # 自定义卷积核，执行模糊操作，也可定义执行锐化操作
    kernel = np.ones([5, 5], np.float32)/25
    dst3 = cv2.filter2D(image, -1, kernel=kernel)
    return cv2.resize(dst3,(settings.x,settings.y)),cv2.resize(image,(settings.x,settings.y))

def read_pic(path):
    for i in os.walk(path):
        root=i[0]
        #return [root+"/"+k for k in i[2]]
        return [root + "/" + k for k in i[2]][:500]


def dataset_make(list_val,list_label):

    dataset = tf.data.Dataset.from_tensor_slices((list_val, list_label))
    dataset = dataset.shuffle(settings.BUFFER_SIZE).prefetch(
        tf.data.experimental.AUTOTUNE).batch(settings.batch)
    return dataset

def makedataset(path):

    val_list=[]
    label_list=[]
    with tqdm(total=len(read_pic(path))) as p_bar:
        for i,j in zip(read_pic(path),range(len(read_pic(path)))):
            val, label = pic_fuzzy(i)
            val_list.append(val / 255.0 + 1e-3)
            label_list.append(label / 255.0 + 1e-3)

            p_bar.update(1)
            p_bar.set_description("Processing {}-th iteration".format(j + 1))
    print("划分数据集")
    x_train, x_test, y_train, y_test = train_test_split(
        val_list, label_list,  # x,y是原始数据
        test_size=0.2  # test_size默认是0.25
    )  # 返回的是 剩余训练集+测试集
    dataset_train = dataset_make(np.array(x_train), np.array(y_train))
    dataset_test = dataset_make(np.array(x_test), np.array(y_test))
    print("数据集划分完成")
    return dataset_train, dataset_test
