import pandas as pd
import numpy as np


def show_information():
    print('================   Matrix Addition and Inner Product ============')
    print('load three matrices from an excel file')
    print('calculate the addition of matrix1 and matrix2')
    print('calculate the inner product of matrix1 and matrix3')
    print('=================================================================')


def read_matrix(path):
    m1 = pd.read_excel(path, sheet_name='Matrix1', header=None).values
    m2 = pd.read_excel(path, sheet_name='Matrix2', header=None).values
    m3 = pd.read_excel(path, sheet_name='Matrix3', header=None).values
    return m1, m2, m3


def show_matrix(matrix1, matrix2, maxtrix3):
    print('Matrix1:')
    print(matrix1)
    print('Matrix2:')
    print(matrix2)
    print('Matrix3:')
    print(maxtrix3)
    print('=================================================================')


def addition(matrix1, matrix2):
    row_left = len(matrix1)  # 左矩阵的行数
    col_left = len(matrix1[0])  # 左矩阵的列数
    row_right = len(matrix2)
    col_right = len(matrix2[0])
    if row_left != row_right or col_left != col_right:
        print('add failed two matrixes don\'t have the same dimension!')
        return None
    result = np.zeros((row_left, col_left))
    for i in range(row_left):
        for j in range(col_right):
            result[i][j] = matrix1[i][j] + matrix2[i][j]

    return result


def show_result(result1, result2):
    print('matrix addition result:')
    print('Matrix4')
    print(result1)
    print('Inner product result:')
    print('Matrix5')
    print(result2)


def vector_inner(v1, v2):
    return np.sum(v1 * v2)


def inner(m1, m3):
    row_left = len(m1)  # 左矩阵的行数
    col_left = len(m1[0])  # 左矩阵的列数
    row_right = len(m3)
    col_right = len(m3[0])
    if col_left != row_right:
        print('dot failed The two matrixes don\'t match, can\'t calculate the product!')
        return None
    result = np.zeros((row_left, col_right))
    for i in range(row_left):
        for j in range(col_right):
            # 左矩阵的第i行乘以右矩阵的第j列
            v1 = m1[i]
            v2 = m3[:, j]
            result[i][j] = vector_inner(v1, v2)
    return result


def save_result(result1, result2):
    df1 = pd.DataFrame(result1)
    df2 = pd.DataFrame(result2)
    writer = pd.ExcelWriter('HW2_result.xlsx')
    df1.to_excel(writer, sheet_name='addition', header=None, index=None)
    df2.to_excel(writer, sheet_name='inner', header=None, index=None)
    writer.save()
    writer.close()


def main():
    show_information()
    matrix1, matrix2, matrix3 = read_matrix(path="Homework2.xlsx")
    show_matrix(matrix1, matrix2, matrix3)
    result1 = addition(matrix1, matrix2)
    result2 = inner(matrix1, matrix3)
    show_result(result1, result2)
    save_result(result1, result2)


if __name__ == "__main__":
    main()
