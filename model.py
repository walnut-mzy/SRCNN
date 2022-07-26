import settings
import tensorflow as tf
from tensorflow.keras.layers import Layer,Conv2D,Input
from tensorflow.keras import Sequential, Model
class SRCNN(Layer):
    def __init__(self):
        super().__init__()

        self.conv1=Conv2D(
    filters=64,
    kernel_size=9,
    padding='same',
    activation=tf.nn.relu
            )
        self.conv2=Conv2D(
    filters=32,
    kernel_size=1,
    padding='same',
    activation=tf.nn.relu
            )
        self.conv3=Conv2D(
    filters=3,
    kernel_size=5,
    padding='same'          # 不设置激活函数
)
    def call(self, inputs):
        x=self.conv1(inputs)
        x=self.conv2(x)
        x=self.conv3(x)
        return x
if __name__ == '__main__':
    inputs = Input(shape=(settings.x,settings.y,3), batch_size=settings.batch)
    SRCNN=SRCNN()
    out = SRCNN(inputs)
    model = Model(inputs=inputs, outputs=out, name='SRCNN-tf2')
    model.summary()