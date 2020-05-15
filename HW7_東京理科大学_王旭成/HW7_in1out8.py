from two_layer_framework import TwoLayFNN
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel('Homework7.xlsx', sheet_name='In1Out8', index_col=0)
data = df.iloc[:, 0].values.reshape(-1, 1)
t = df.iloc[:, 1:].values

twolayFNN_M = TwoLayFNN(input_size=1, hidden_size=30, output_size=8)
twolayFNN_M.train(data, t, epoch=100, batch_size=20, optimizer='Momentum', m=0.9)

plt.figure(figsize=(16, 9))
for i in range(8):
    plt.subplot(2, 4, i + 1)
    plt.title('in1out8_{0}'.format(i + 1))
    plt.xlabel('X')
    plt.ylabel('T{0}'.format(i + 1))
    plt.scatter(data, t[:, i], s=2)
    plt.scatter(data, twolayFNN_M.predict(data)[:, i], s=2)

plt.show()
