import numpy as np
import pandas as pd

""""
感知机类说明
------------------------------------------------------
私有属性 : 权重值w 学习率n 最大学习次数Max_iteration 最好状态best_w。
私有方法 : update存放w的更新法则
-----------------------------------------------------
构造方法 :
用户可以对学习率，初始权值w，最大学习次数iteration进行指定，在未指定的情况下默认值如下
n=1, W=np.array([0, 0, 1.1]), Max_iteration=50
-----------------------------------------------------
公有方法 :
.score()                |    得到最优解时猜错的个数
.train()                |    对于给定的x与t进行学习
.calculate_accuracy()   |    计算给定w下的误差x的预测值与t的误差
"""


class Perceptron(object):
    def __init__(self, n=1, W=np.array([0, 0, 1.1]), Max_iteration=50):
        self.__n = n
        self.__W = W
        self.__Max_iteration = Max_iteration
        self.__best_W = None
        self.__loss_score = None

    def train(self, X_train, T_train):
        self.__loss_score = self.calculate_accuracy(self.__W, X_train, T_train)  # 计算初始值w的猜错个数
        self.__best_W = self.__W  # 将最优解w先预设为初始w
        print('loss : {0}'.format(self.__loss_score))
        if self.__loss_score == 0:
            print('你的运气也太好了吧！一上来就全猜对了')
            print('当前的权重w的值为:')
            print(self.__best_W)
            return None

        for i in range(self.__Max_iteration):
            each_line_X = X_train[i % X_train.shape[0]]
            net = np.dot(self.__W, each_line_X)
            y = 1 if net > 0 else 0
            t = T_train[i % X_train.shape[0]]
            # 如果预测错误
            if y != t:
                sign, self.__W = self.__update(self.__W, y, t, each_line_X)
                temp_loss = self.calculate_accuracy(self.__W, X_train, T_train)
                if temp_loss < self.__loss_score:
                    self.__loss_score = temp_loss
                    self.__best_W = self.__W
                print('Date : {0}  Modify Weight '.format(i % 5) + sign)
                print('current loss is ' + str(temp_loss))
                print(each_line_X)

                # 如果更新之后的w已经实现了100%正确率，则中断程序
                if self.__loss_score == 0:
                    return

            else:
                print('Date : {0}  OK'.format(i % 5))
                print(each_line_X)
            if i % 5 == 4:
                print('=====================')
            # 如果预测正确

    def calculate_accuracy(self, __W, X_train, T_train):
        count = 0  # 猜错的个数
        for i in range(X_train.shape[0]):
            net = np.dot(__W, X_train[i])
            y = 1 if net > 0 else 0
            if y != T_train[i]:
                count = count + 1
        return count

    def get_best_w(self):
        return self.__best_W

    # 更新参数w的函数update，我并不希望用户知道它的存在，所以设置为私有方法
    def __update(self, __W, y, t, each_line_X):
        if t == 1 and y == 0:
            return '+', __W + self.__n * each_line_X
        else:
            return '-', __W - self.__n * each_line_X

    def score(self):
        return self.__loss_score


def read_date(path):
    df1 = pd.read_excel('HomeWork4.xlsx', sheet_name=1)
    df1.iloc[:, 0] = 1
    X = df1.iloc[:, 0:3]  # 得到前3列特征值
    T = df1.iloc[:, 3]  # 得到标签值
    return X.values, T.values


def main():
    X_train, T_train = read_date(path='Homework4')
    perceptron = Perceptron(Max_iteration=500)  # 实例化一个感知机
    perceptron.train(X_train, T_train)
    print(perceptron.get_best_w())  # 取得训练后的最佳权重
    print(perceptron.score())  # 显示最优情况下有几个猜错


if __name__ == '__main__':
    main()
