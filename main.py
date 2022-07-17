# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


from train import train
from predict import predict
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    mod=predict
    path=""
    if mod=="train":
        train()
    elif mod=="predict":
        predict(path)

