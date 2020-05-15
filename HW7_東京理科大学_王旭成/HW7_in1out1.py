from two_layer_framework import TwoLayFNN
import pandas as pd
import matplotlib.pyplot as plt


def draw_picture(model_name, mse, data, t, twolayFnn):
    plt.figure()
    plt.title(model_name)
    plt.xlabel('X')
    plt.ylabel('T')
    plt.scatter(data.reshape(1, -1), t.reshape(1, -1), c='blue')
    plt.scatter(data.reshape(1, -1), twolayFnn.predict(data).reshape(1, -1), c='orange')
    plt.text(-10, 3, 'MSE : error' + mse[0:5])
    plt.savefig(model_name + '_result.png')


# 读取数据
df = pd.read_excel('Homework7.xlsx', sheet_name=0, index_col=0)
data = df.iloc[:, 0].values.reshape(-1, 1)
t = df.iloc[:, 1].values.reshape(-1, 1)
# 取出了所有的数据与标签
# 将前800个数据作为学习集，而后面的数据作为测试集
x_train = data[0:800]
t_train = t[0:800]
x_test = data[800:]
t_test = t[800:]
# 1-5-1结构,训练30个epoch,每次抓20笔数据
twolayFNN = TwoLayFNN(input_size=1, hidden_size=5, output_size=1)
twolayFNN.train(x_train, t_train, epoch=30, batch_size=20, optimizer='SGD')
mse = str(twolayFNN.loss(x_test, t_test))
print('在未知测试数据中的表现 : ')
print('MSE loss of SGD : ' + mse)
draw_picture('SGD', mse, data, t, twolayFNN)
print('-------------------------------------------------------------------------------')
twolayFNN_M = TwoLayFNN(input_size=1, hidden_size=5, output_size=1)
twolayFNN_M.train(x_train, t_train, epoch=30, batch_size=20, optimizer='Momentum', m=0.9)
mse = str(twolayFNN_M.loss(x_test, t_test))
print('在未知测试数据中的表现 : ')
print('MSE loss of Momentum : ' + mse)
draw_picture('Momentum', mse, data, t, twolayFNN_M)
print('-------------------------------------------------------------------------------')
twolayFNN_AG = TwoLayFNN(input_size=1, hidden_size=5, output_size=1)
twolayFNN_AG.train(x_train, t_train, epoch=30, batch_size=20, optimizer='AdaGrad')
mse = str(twolayFNN_AG.loss(x_test, t_test))
print('在未知测试数据中的表现 : ')
print('MSE loss of AdaGrad : ' + mse)
draw_picture('AdaGrad', mse, data, t, twolayFNN_AG)
