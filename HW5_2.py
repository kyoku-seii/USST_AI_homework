import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
Adaline类说明:
类中具有私有属性学习率n，初始权重w，最大学习次数Max_iteration,更新一次w所用的数据量batch_size，最佳权重best_W
最佳mse误差mse，当前mse误差current_mse。
之所以把这些属性全部设置为私有，因为我不希望用户可以通过"Adaline.mse=0"的方式直接对误差进行修改从而作弊，破坏类的封装性
私有方法: __SGD:保存了梯度下降的运算公式   __loss_function:计算MSE误差 
对外API
train()           |  学习方法
get_best_weight() |  返回学习过程中的最佳权重
best_score()      |  返回学习过程中最小的mse误差
score()           |  返回模型最后的分数
predict()         |  以最后的w值进行预测 该接口支持泛型，可传入特定数字或者数组
save_bestW()      |  将最优w保存到一个excel中
"""


class Adaline(object):
    # 用户可以自己定义学习率n，初始权重W,最大学习次数iteration，每次学习采用的样本数batch_size
    # 如果用户在新建类对象时，不传入任何参数，则默认值如下
    def __init__(self, n=0.01, W=np.array([10, -1]), Max_iteration=50, batch_size=5):
        self.__n = n
        self.__W = W
        self.__Max_iteration = Max_iteration
        self.__batch = 5
        self.__best_W = None  # 初始化最佳权重best_W
        self.__mse = None  # 初始化最佳mse误差
        self.__current_mse = None

    def train(self, X_train, T_train):
        self.__best_W = self.__W
        self.__mse = self.__loss_function(X_train, T_train, np.arange(X_train.shape[0]), self.__W)
        plt.figure(figsize=(8, 6), dpi=80)
        # plt.ion()
        for i in range(self.__Max_iteration):

            # 如果用户传入的batch_size过大 都超过样本数的情况下，将batch_size变为样本数
            if self.__batch > X_train.shape[0]:
                self.__batch = X_train.shape[0]
                # 理想情况下循环X_train.shape[0]/self.__batch回数下，检查过了所有数据，记iteration+1
            for j in range(int(X_train.shape[0] / self.__batch)):
                # 随机从样本中取出batch_size 个数来组成一个batch
                index = np.random.randint(X_train.shape[0], size=self.__batch)
                self.__W = self.__SGD(self.__W, self.__n, T_train, index, X_train)  # 对w进行更新
                self.__current_mse = self.__loss_function(X_train, T_train, np.arange(X_train.shape[0]),
                                                          self.__W)  # 得到当前总体mse
                if self.__current_mse < self.__mse:
                    self.__mse = self.__current_mse
                    self.__best_W = self.__W
            self.__current_mse = str(self.__loss_function(X_train, T_train, np.arange(X_train.shape[0]), self.__W))
            print('iteration : {0}/{1}'.format(i + 1, self.__Max_iteration))
            print('MSE loss: ' + self.__current_mse)
            print(self.__W)
            print('-----------------------------------------------------')

            self.__draw_picture(X_train, T_train, self.__current_mse)
        # plt.ioff()
        plt.show()

    def __SGD(self, w, n, t, index, x):
        number = len(index)
        sum = 0
        for i in index:
            sum = sum + (t[i] - np.sum(x[i] * w)) * x[i]
        return w + 2 * n * sum / number

    def __loss_function(self, X, T, index, W):
        n = len(index)
        sum = 0
        for i in index:
            sum = sum + (T[i] - np.sum(X[i] * W)) ** 2
        return sum / n

    def get_best_weight(self):
        return self.__best_W

    def best_score(self):
        return self.__mse

    def score(self):
        return self.__current_mse

    def predict(self, x):
        if isinstance(x, np.ndarray) or isinstance(x, list):
            x = np.array(x)
            x0_col = np.tile(1, len(x))
            matrix = np.concatenate((x0_col.reshape(-1, 1), x.reshape(-1, 1)), axis=1)
            y = np.dot(matrix, self.__W, )
            return y
        if isinstance(x, float) or isinstance(x, int):
            matrix = np.array([1, x])
            y = np.dot(matrix, self.__W, )
            return y
        raise TypeError('请输入一个数或者列表或者array')

    def __draw_picture(self, X_train, T_train, loss):
        plt.cla()
        plt.scatter(X_train[:, 1], T_train, c='darkorange', edgecolors='black')
        x_min = np.min(X_train[:, 1])
        x_max = np.max(X_train[:, 1])
        interval = (x_max - x_min) / 10
        y = self.predict(np.arange(x_min, x_max + interval, interval))
        plt.plot(np.arange(x_min, x_max + interval, interval), y, c='r', linewidth=3)  # 直接调用predict接口减少代码量
        plt.title('Adaline')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.text(x_min, np.max(T_train)+0.1, 'MSE error:' + loss[0:5], fontdict={'size': '10', 'color': 'black'})
        plt.pause(0.3)
        plt.show()

    def save_bestW(self, path, sheet_name='Adaline liner_result'):
        df = pd.DataFrame(self.__best_W.reshape(1, 2))
        df.columns = ['bias', 'Weight']
        write = pd.ExcelWriter(path)
        df.to_excel(write, sheet_name=sheet_name, index=False)
        write.save()


def read_date(path, sheet_name):
    df1 = pd.read_excel(path, sheet_name=sheet_name)
    df1.iloc[:, 0] = 1
    X = df1.iloc[:, 0:2]  # 得到前2列特征值
    T = df1.iloc[:, 2]  # 得到标签值
    return X.values, T.values


def main():
    X_train, T_train = read_date(path='Homework5.xlsx', sheet_name='HW_5_MP_Linear')
    adaline = Adaline()  # new一个Adaline对象
    adaline.train(X_train, T_train)  # 带入数据学习
    print(adaline.get_best_weight())
    print(adaline.best_score())  # 在整个学习过程中的最高分
    print(adaline.score())  # 在学习结束之后此时模型的分数
    print(adaline.predict(np.array([1, 2, 3])))  # 预测x=1,x=2,x=3时候的值
    print(adaline.predict([1, 2, 3]))  # 如果用户想输入一个list 或者单个数值，接口也能够实现预测
    print(adaline.predict(5))
    adaline.save_bestW(path='H5_Line_result.xlsx', sheet_name='homework_2')


if __name__ == '__main__':
    main()
