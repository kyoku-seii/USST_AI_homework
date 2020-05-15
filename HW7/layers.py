import numpy as np


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size


def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))


class Layer(object):
    def __init__(self, W):
        self.W = W
        self.x = None
        # 用来保存反向传播的梯度结果
        self.dw = None

    def forward(self, x):
        # 对x进行修改，添加b项
        b = np.ones((x.shape[0], 1))
        x = np.concatenate([b, x], 1)
        self.x = x
        output = np.dot(x, self.W)
        return output

    def backward(self, dout):
        self.dw = np.dot(self.x.T, dout)
        dx = np.dot(dout, self.W.T)
        dback = dx[:, 1:]  # 第一列为b的dback,是不需要向前传播的。
        return dback


class Sigmoid(object):
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx


class MseLoss(object):
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = x
        self.loss = np.mean((self.t - self.y) ** 2)
        return self.loss

    def backward(self, dout=1):
        dx = -2 * (self.t - self.y) / self.t.shape[0]
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size

        return dx
