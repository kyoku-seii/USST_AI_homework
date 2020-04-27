import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_date(path, sheet_name):
    df1 = pd.read_excel(path, sheet_name=sheet_name)
    df1.iloc[:, 0] = 1
    X = df1.iloc[:, 0:3]  # 得到前3列特征值
    T = df1.iloc[:, 3]  # 得到标签值
    T.loc[T == -1] = 0  # 将所有标记为-1的点改为0
    return X.values, T.values


class AdalineClassification(object):
    def __init__(self, n=0.1, W=np.array([-1.2, -0.1, 1.009]), Max_iteration=15, batch_size=5):
        self.__n = n
        self.__W = W
        self.__Max_iteration = Max_iteration
        self.__batch = 5
        self.__best_W = None  # cross_entropy最低时的权重
        self.__best_loss = None  # cross_entropy的最低值
        self.__current_loss = None  # 模型当前的loss
        self.__accuracy = None

    def train(self, X_train, T_train):
        self.__best_W = self.__W
        self.__best_loss = self.__loss_function(X_train, T_train, self.__W)
        plt.figure(figsize=(8, 6), dpi=80)
        for i in range(self.__Max_iteration):
            if self.__batch > X_train.shape[0]:
                self.__batch = X_train.shape[0]
                # 理想情况下循环X_train.shape[0]/self.__batch回数下，检查过了所有数据，记iteration+1
            for j in range(int(X_train.shape[0] / self.__batch)):
                # 随机从样本中取出batch_size 个数来组成一个batch
                index = np.random.randint(X_train.shape[0], size=self.__batch)
                self.__W = self.__Gradient_descent(self.__W, self.__n, T_train, index, X_train)
                self.__current_loss = self.__loss_function(X_train, T_train, self.__W)
                if self.__current_loss < self.__best_loss:
                    self.__best_loss = self.__current_loss
                    self.__best_W = self.__W
            self.__current_loss = str(self.__loss_function(X_train, T_train, self.__W))
            self.__accuracy = str(self.__calculate_accuray(self.__W, T_train, X_train))
            print('iteration : {0}/{1}'.format(i + 1, self.__Max_iteration))
            print('cross_entropy loss: ' + self.__current_loss)
            print('accuracy: ' + self.__accuracy)
            print(self.__W)
            print('-----------------------------------------------------')
            self.__draw_picture(X_train, T_train, self.__accuracy)
        plt.show()

    def __loss_function(self, X_train, T_train, __W):
        n = X_train.shape[0]
        sum = 0
        for i in range(n):
            sum = sum + -(T_train[i] * np.log(self.__predict_function(X_train[i])) + (1 - T_train[i]) * np.log(
                1 - self.__predict_function(X_train[i])))
        return sum / n

    def __predict_function(self, X):
        z = np.sum(np.dot(self.__W, X))
        return self.__sigmoid(z)

    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def __Gradient_descent(self, __W, __n, T_train, index, X_train):
        number = len(index)
        sum = 0
        for i in index:
            sum = sum + (self.__predict_function(X_train[i]) - T_train[i]) * X_train[i]
        return __W - (__n * sum / number)

    def __calculate_accuray(self, __W, T_train, X_train):
        correct = 0
        for i in range(T_train.shape[0]):
            net = np.dot(__W, X_train[i])
            y = 1 if net > 0 else 0
            if y == T_train[i]:
                correct += 1
        return correct / T_train.shape[0]

    def __draw_picture(self, X_train, T_train, accuracy):
        plt.cla()
        x_min = np.min(X_train[:, 1])
        x_max = np.max(X_train[:, 1])
        interval = (x_max - x_min) / 10
        X = np.arange(x_min, x_max + interval, interval)
        Y = []
        for x in X:
            y = - (self.__W[0] * 1 + self.__W[1] * x) / self.__W[2]
            Y.append(y)

        plt.plot(X, Y, c='r', linewidth=2)
        plt.title('AdalineClassification')
        plt.xlabel('X')
        plt.ylabel('Y')
        for i in range(T_train.shape[0]):
            if T_train[i] == 0:
                plt.scatter(X_train[i, 1], X_train[i, 2], c='red', edgecolors='black')
            else:
                plt.scatter(X_train[i, 1], X_train[i, 2], c='blue', edgecolors='black')
        plt.text(x_max - 0.5, np.max(T_train), 'accuracy:' + accuracy[0:5], fontdict={'size': '10', 'color': 'black'})
        plt.pause(0.3)

    def predict(self, coordinate):
        x = [1] + coordinate
        net = np.dot(self.__W, np.array(x))
        group = 1 if net > 0 else -1
        return group

    def get_bestW(self):
        return self.__best_W

    def get_best_score(self):
        return self.__best_loss

    def get_score(self):
        return self.__current_loss

    def get_accuracy(self):
        return self.__accuracy


def main():
    X_train, T_train = read_date(path='Homework5.xlsx', sheet_name='HW_5_CP')
    ac = AdalineClassification()
    ac.train(X_train, T_train)
    print(ac.predict([0, 0]))  # 预测给定的坐标点属于哪一类返回1或者-1
    print(ac.get_bestW())  # 得到训练过程中cross_entropy最小时候的w值
    print(ac.get_best_score())  # 得到当时的具体cross_entropy的大小
    print(ac.get_score())  # 得到模型现在的cross_entropy大小
    print(ac.get_accuracy())  # 得到模型现在的分类准确度


if __name__ == '__main__':
    main()
