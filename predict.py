import settings
from model import SRCNN
import cv2

from PIL import Image
def predict(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.IMREAD_COLOR)
    model=SRCNN()
    model=model.build()
    model.load_weights(settings.weight_path)
    image=image/255+0.001
    # 预测

    result = model.predict(image)
    image = Image.fromarray((result-0.01)*255)  # 将之前的矩阵转换为图片
    image.show()  # 调用本地软件显示图片，win10是叫照片的工具