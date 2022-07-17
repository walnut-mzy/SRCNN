# SRCNN

**#一些参数**

```python
import os
BUFFER_SIZE=500
batch=8
save_path="model_save"
datapath="X2"
logdir="logs/"
#设置图片高低
x=512
y=512
if not os.path.exists(logdir):
        os.mkdir(logdir)
if not os.path.exists(save_path):
        os.mkdir(save_path)
epoch=100
initial_epoch=2
loss="MSE"
lr=1e-5
gpu=True
weight_path="./MODEL"
```

模型运行只需要配置好环境，在main.py文件运行即可。

#训练集和测试集被保存在百度网盘中
链接：https://pan.baidu.com/s/1i2_2yPxjaA4IWNPK8BKK7g?pwd=jlnx 
提取码：jlnx
