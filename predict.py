import settings
from model import SRCNN
import cv2
import tensorflow as tf
from PIL import Image
from tensorflow.keras.layers import Input
import numpy as np
from tensorflow.keras import Sequential, Model
def predict(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.IMREAD_COLOR)
    image=cv2.resize(image,(settings.x,settings.y))
    # 此为均值模糊
    # （30,1）为一维卷积核，指在x，y方向偏移多少位
    image = cv2.blur(image, (30, 1))

    # 此为中值模糊，常用于去除椒盐噪声
    image = cv2.medianBlur(image, 15)

    # 自定义卷积核，执行模糊操作，也可定义执行锐化操作
    kernel = np.ones([5, 5], np.float32) / 25
    image = cv2.filter2D(image, -1, kernel=kernel)
    inputs = Input(shape=(settings.x, settings.y, 3))
    model1 = SRCNN()
    out = model1(inputs)
    model = Model(inputs=inputs, outputs=out, name='SRCNN-tf2')
    model.load_weights("model_save/transformers100.h5")
    image1=image
    image=image/255+0.001
    image=tf.expand_dims(image,axis=0)
    result = model.predict(image)
    result=tf.squeeze(result,axis=0)

    result=np.round(result.numpy())
    print(result.shape)
    cv2.imshow("x1",result)
    cv2.waitKey(0)
    print(tf.image.psnr(result,image1,255))
predict("X2/000002x2.png")