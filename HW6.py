import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class FeedforwardNet:
    def __init__(self, hidden_size=2, ):
        self.__input_size = 1
        self.hidden_size = hidden_size
        self.__output_size = 1
        self.w1 = None
        self.w2 = None
        self.hs = self.__dict__  # 利用魔法方法生成hidden_size个变量
        for i in range(hidden_size):
            self.hs['h' + str(i + 1)] = None

    def init_w_from_excel(self, path='Homework6.xlsx'):
        df1 = pd.read_excel(path, sheet_name=0, index_col=0)
        self.w1 = df1.values
        print('Weight in Hid:')
        print(self.w1)
        print('')
        df2 = pd.read_excel(path, sheet_name=1, index_col=0)
        self.w2 = df2.values
        print('Weight in Out:')
        print(self.w2)
        print('')

    def forecast(self, x_train):
        x1 = [1, x_train]  # 第一层的x1
        h = self.__sigmoid(np.dot(x1, self.w1))  # 隐藏层的h
        for i in range(h.shape[0]):
            self.hs['h' + str(i + 1)] = h[i]  # 分别记录每个node的值
        x2 = np.concatenate((np.array([1]), h), axis=0)
        y = np.dot(x2, self.w2)
        return y[0]

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def init_w(self):
        self.w1 = np.random.rand(2, self.hidden_size)
        self.w2 = np.random.rand(self.hidden_size + 1, 1)


def show_information():
    print('========== Feedforward Netural Network ============')
    print('Load the weights from an Excel file.')
    print('Input a data x and output the ralatived forecasting value y.')
    print('========== Feedforward Netural Network ============')
    print('')


def show_picture(net):
    x = np.arange(-12, 12, 1)
    h1 = []
    h2 = []
    y = []
    for i in x:
        y.append(net.forecast(i))
        h1.append(net.h1)
        h2.append(net.h2)
    plt.subplot(311)
    plt.plot(x, h1)
    plt.ylabel('h1')
    plt.subplot(312)
    plt.plot(x, h2, c='green')
    plt.ylabel('h2')
    plt.subplot(313)
    plt.plot(x, y, c='red')
    plt.ylabel('y', )
    plt.show()


def main():
    show_information()
    net = FeedforwardNet()
    net.init_w_from_excel('Homework6.xlsx')
    x_train = float(input('请输入x信号的输入值  '))
    y = net.forecast(x_train)
    print('输出y为: {0}'.format(y))
    show_picture(net)

    # ==================================================
    print('====  进入指定隐藏层部分  =====')
    hidden_size = int(input('请输入隐藏层node的数量'))
    if hidden_size <= 0:
        print('输入正确的大小')
        return
    net2 = FeedforwardNet(hidden_size)
    net2.init_w()  # 随机生成参数w
    x_train = float(input('请输入x信号的输入值  '))
    y2 = net2.forecast(x_train)
    print('输出y为: {0}'.format(y2))


if __name__ == '__main__':
    main()
