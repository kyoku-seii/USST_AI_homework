import pandas as pd
from two_layer_classification import TwoLay_classification
import matplotlib.pyplot as plt

df = pd.read_excel('Homework7.xlsx', sheet_name='Classification', index_col=0)
x = df.iloc[:, 0:2].values
t = df.iloc[:, 2:].values

x_train = x[0:2400, :]
t_label = t[0:2400, :]
x_test = x[2400:, :]
t_test = t[2400:, :]

twolay_calssification = TwoLay_classification(input_size=2, hidden_size=5, output_size=2)
twolay_calssification.train(x_train, t_label, batch_size=30, epoch=50, optimizer='Momentum', m=0.9)
result = twolay_calssification.predict(x_test)
print('识别率')
acccuracy = twolay_calssification.accuracy(x_test, t_test)
print(acccuracy)

plt.figure()
for i in range(x_train.shape[0]):
    if t_label[i, 0] > t_label[i, 1]:
        plt.scatter(x_train[i, 0], x_train[i, 1], c='', marker='o', s=2, edgecolors='red')
    else:
        plt.scatter(x_train[i, 0], x_train[i, 1], c='', marker='o', s=2, edgecolors='blue')
for j in range(x_test.shape[0]):
    if result[j, 0] > result[j, 1]:
        plt.scatter(x_test[j, 0], x_test[j, 1], c='black', marker='x', s=10)
    else:
        plt.scatter(x_test[j, 0], x_test[j, 1], c='black', marker='^', s=10)

plt.show()
