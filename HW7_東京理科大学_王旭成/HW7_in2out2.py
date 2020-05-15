from two_layer_framework import TwoLayFNN
import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_excel('Homework7.xlsx', sheet_name=2, index_col=0)
data = df.iloc[:, 0].values.reshape(-1, 1)
t = df.iloc[:, 1:].values
# 取出了所有的数据与标签
twolayFNN = TwoLayFNN(input_size=1, hidden_size=30, output_size=2)
twolayFNN.train(data, t, epoch=50,batch_size=30)
