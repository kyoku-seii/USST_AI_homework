from two_layer_framework import TwoLayFNN
import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_excel('Homework7.xlsx', sheet_name='In1Out2', index_col=0)
data = df.iloc[:, 0].values.reshape(-1, 1)
t = df.iloc[:, 1:].values
# 取出了所有的数据与标签
# 1-10-2
twolayFNN_M = TwoLayFNN(input_size=1, hidden_size=6, output_size=2)
twolayFNN_M.train(data, t, epoch=30, batch_size=20, optimizer='Momentum', m=0.9)

plt.figure()
plt.subplot(2,1,1)
plt.title('in1out2_1')
plt.xlabel('X')
plt.ylabel('T1')
plt.scatter(data, t[:, 0], c='blue',s=3)
plt.scatter(data, twolayFNN_M.predict(data)[:,0], c='orange',s=3)
plt.subplot(2,1,2)
plt.title('in1out2_2')
plt.xlabel('X')
plt.ylabel('T2')
plt.scatter(data, t[:, 1], c='blue',s=3)
plt.scatter(data, twolayFNN_M.predict(data)[:,1], c='lightgreen',s=3)
plt.show()

