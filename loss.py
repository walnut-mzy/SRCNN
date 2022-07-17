import tensorflow as tf
def loss_cos(x,y):
    # print(type(x))
    # x=tf.reduce_mean(tf.square(x-y))
    # print(x.shape)
    # return x
    number = tf.convert_to_tensor(x, dtype=tf.float32)+1
    number1 = tf.convert_to_tensor(y, dtype=tf.float32)+1
    number = tf.math.log(number)+1e-5
    number_log = number / (number + 1)

    x = (1 - number1) * (1 - number_log)
    z = tf.math.log(number_log) * number1
    loss1 = -(x + z)

    x=tf.reduce_sum(loss1,axis=-1)
    X=tf.reduce_sum(x,axis=-1)
    print(x)
    return x
