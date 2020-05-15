import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from layers import *
from collections import OrderedDict
from optimizer import *


class TwoLayFNN:
    def __init__(self, input_size, hidden_size, output_size, weight_area=0.6):
        # 利用哈希表存放各层的权重,初始权重范围为(-0.3,-0.3)
        self.params = {'W1': weight_area * np.random.rand(input_size + 1, hidden_size) - 0.3,
                       'W2': weight_area * np.random.rand(hidden_size + 1, output_size) - 0.3}
        # 按照顺序形成各层
        self.layers = OrderedDict()  # 从collections中导入顺序哈希表
        self.layers['lay1'] = Layer(self.params['W1'])
        self.layers['Sigmoid'] = Sigmoid()
        self.layers['lay2'] = Layer(self.params['W2'])
        self.lastLayer = MseLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def gradient(self, x, t):
        self.loss(x, t)
        dout = self.lastLayer.backward()
        layers = list(self.layers.values())
        layers.reverse()  # 倒转顺序
        for layer in layers:
            dout = layer.backward(dout)
        grads = {'W1': self.layers['lay1'].dw, 'W2': self.layers['lay2'].dw}
        return grads

    def train(self, x, t, batch_size=100, learning_rate=0.1, epoch=10, optimizer='SGD', m=0.9):
        iter_per_epoch = int(x.shape[0] / batch_size)
        cur_epoch = 0
        while cur_epoch < epoch:
            for i in range(iter_per_epoch):
                batch_index = np.random.choice(x.shape[0], batch_size)
                x_batch = x[batch_index]
                t_batch = t[batch_index]
                grad = self.gradient(x_batch, t_batch)
                if optimizer == 'SGD':
                    optimizer = SGD(lr=learning_rate)
                elif optimizer == 'Momentum':
                    optimizer = Momentum(momentum=m)
                elif optimizer == 'AdaGrad':
                    optimizer = AdaGrad(lr=learning_rate)
                optimizer.update(self.params, grad)
            print("Epoch {0}/{1}".format(cur_epoch + 1, epoch))
            print("loss : " + str(self.loss(x, t)))
            cur_epoch += 1
