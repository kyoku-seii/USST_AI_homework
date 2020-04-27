import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from HW5_2 import Adaline


def read_date(path, sheet_name):
    df1 = pd.read_excel(path, sheet_name=sheet_name)
    df1.iloc[:, 0] = 1
    X = df1.iloc[:, 0:2]  # 得到前2列特征值
    T = df1.iloc[:, 2]  # 得到标签值
    return X.values, T.values


def main():
    X_train, T_train = read_date(path='Homework5.xlsx', sheet_name='HW_5_MP_un-Linear')
    adaline = Adaline()  # new一个Adaline对象
    adaline.train(X_train, T_train)  # 带入数据学习
    adaline.save_bestW(path='HW5_unliner_result.xlsx', sheet_name='un-Linear_result')
    print(adaline.get_best_weight())
    print(adaline.best_score())


if __name__ == '__main__':
    main()
