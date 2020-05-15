from layers import *
from optimizer import *
from two_layer_framework import TwoLayFNN


# 让分类模型继承回归模型，将最后一层改为softmax。并创建父类所没有的识别精度accura方法

class TwoLay_classification(TwoLayFNN):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__(input_size, hidden_size, output_size)
        self.lastLayer = SoftmaxWithLoss()

    def accuracy(self, x, t):
        y_val = self.predict(x)
        y = np.argmax(y_val, axis=1)
        t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / x.shape[0]
        return accuracy
