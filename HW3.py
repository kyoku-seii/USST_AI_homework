import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    y = 1 / (1 + np.exp(-x))
    return y


def nordis(x, mean, std):  # 建立正态分布函数
    return 1 / (np.sqrt(2 * np.pi) * std) * np.exp(-0.5 * (((x - mean) / std) ** 2))


def draw_function(mean, std):
    plt.figure(figsize=(8, 5), dpi=80)
    plt.subplot(1, 2, 1)
    x = np.arange(-10.0, 10.0, 0.1)
    y = sigmoid(x)
    plt.plot(x, y, color='blue', linewidth=3)
    plt.xlim(-6, 6)
    plt.ylim(-0.5, 1.5)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title("sigmoid function")
    plt.subplot(1, 2, 2)
    y2 = nordis(x, mean, std)
    plt.plot(x, y2, color="red", linewidth=3)
    plt.xlim(-3, 3)
    plt.ylim(0, 1)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title("Normal Distribution")
    plt.show()


def main():
    print('请输入正态分布的均值')
    mean = float(input('mean:'))
    print('请输入正态分布的方差')
    std = float(input('std:'))
    draw_function(mean, std)


if __name__ == '__main__':
    main()
