import tensorflow as tf
class PSNR(tf.keras.metrics.Metric):
    # 计算正确预测的个数
    def __init__(self, name='psnr', **kwargs):
        super(PSNR, self).__init__(name=name, **kwargs)
        self.score=0
        self.count=0
    def update_state(self, y_true, y_pred, sample_weight=None):
        psnr=tf.image.psnr(
            (y_true-1e-3)*255,
            (y_pred-1e-3)*255,
            255,
            name=None
        )
        self.score+=psnr
        self.count+=1

    def result(self):
        return self.score/self.count

    def reset_states(self):
        self.tp.assign(0.)