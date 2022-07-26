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
model="SRCNN"