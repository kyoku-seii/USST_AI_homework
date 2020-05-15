from two_layer_framework import TwoLayFNN
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel('Homework7.xlsx', sheet_name='In1Out4', index_col=0)
data = df.iloc[:, 0].values.reshape(-1, 1)
t = df.iloc[:, 1:].values

twolayFNN_M = TwoLayFNN(input_size=1, hidden_size=20, output_size=4)
twolayFNN_M.train(data, t, epoch=50, batch_size=20, optimizer='Momentum', m=0.9)

plt.figure()
plt.subplot(2, 2, 1)
plt.title('in1out4_1')
plt.xlabel('X')
plt.ylabel('T1')
plt.scatter(data, t[:, 0], c='blue', s=3)
plt.scatter(data, twolayFNN_M.predict(data)[:, 0], c='orange', s=3)
plt.subplot(2, 2, 2)
plt.title('in1out4_2')
plt.xlabel('X')
plt.ylabel('T2')
plt.scatter(data, t[:, 1], c='blue', s=3)
plt.scatter(data, twolayFNN_M.predict(data)[:, 1], c='lightgreen', s=3)
plt.subplot(2, 2, 3)
plt.title('in2out4_3')
plt.xlabel('X')
plt.ylabel('T3')
plt.scatter(data, t[:, 2], c='blue', s=3)
plt.scatter(data, twolayFNN_M.predict(data)[:, 2], c='salmon', s=3)
plt.subplot(2, 2, 4)
plt.title('in1out4_4')
plt.xlabel('X')
plt.ylabel('T4')
plt.scatter(data, t[:, 3], c='blue', s=3)
plt.scatter(data, twolayFNN_M.predict(data)[:, 3], c='deeppink', s=3)
plt.show()

