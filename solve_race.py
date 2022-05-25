import math
import copy
from timeit import Timer
import matplotlib.pyplot as plt


class Matrix:   # 矩阵类
    def __init__(self, row, col, value=0):
        self.size = (row, col)      # 矩阵的大小
        self.row = row              # 矩阵的行数
        self.col = col              # 矩阵的列数
        self.matrix = [[value for _ in range(col)] for _ in range(row)]     # 默认创建零矩阵

    def __getitem__(self, item):    # 返回相应位置的值
        if isinstance(item, int):
            if self.col > 1:
                return self.matrix[item - 1]    # 返回矩阵的第i行
            elif self.col == 1:
                return self.matrix[item - 1][0]    # 返回列向量的第i个分量
        elif isinstance(item, tuple):
            return self.matrix[item[0] - 1][item[1] - 1]    # 返回(i, j)元的值

    def __setitem__(self, key, value):  # 设置(i, j)元的值
        if isinstance(key, int):
            if self.col > 1:
                self.matrix[key - 1] = copy.deepcopy(value)   # 设置矩阵的第i行
            elif self.col == 1:
                self.matrix[key - 1][0] = value     # 设置列向量的第i个分量
        elif isinstance(key, tuple):
            self.matrix[key[0] - 1][key[1] - 1] = value

    def exchange(self, i, j):   # 交换矩阵的i行和j行
        r = self[i]
        self[i] = self[j]
        self[j] = r

    def __str__(self):      # 直接输出的效果
        string = '[ '
        if self.row == 1 and self.col == 1:     # 只有一个矩阵元
            string += f'{self[1, 1]} ]'
        elif self.row == 1 and self.col > 1:      # 只有一行
            for c in range(1, self.col):
                string += f'{str(self[1, c])}  '
            string += f'{str(self[1, self.col])} ]'
        elif self.row > 1 and self.col == 1:      # 只有一列
            for r in range(1, self.row):
                string += f'{str(self[r, 1])}\n  '
            string += f'{str(self[self.row, 1])} ]'
        else:
            for r in range(1, self.row):
                for c in range(1, self.col):
                    string += f'{str(self[r, c])}  '
                string += f'{str(self[r, self.col])}\n  '
            for c in range(1, self.col):
                string += f'{str(self[self.row, c])}  '
            string += f'{str(self[self.row, self.col])} ]'
        return string


def cho(A:Matrix):     # 对称正定矩阵的Cholesky分解
    for j in range(1, A.col + 1):
        for k in range(1, j):
            A[j, j] -= A[j, k] ** 2
        A[j, j] = math.sqrt(A[j, j])            # 开平方根
        for i in range(j + 1, A.row + 1):
            for k in range(1, j):
                A[i, j] -= A[i, k] * A[j, k]
            A[i, j] /= A[j, j]
    return A

def cho2(A:Matrix):     # 对称正定矩阵的Cholesky分解-2
    for i in range(1, A.row + 1):
        for j in range(1, i):
            for k in range(1, j):
                A[i, j] -= A[i, k] * A[k, k] * A[j, k]      # 多次乘d(i)
            A[i, j] /= A[j, j]
        for k in range(1, i):
            A[i, i] -= A[i ,k] ** 2 * A[k, k]
    return A

def cho3(A:Matrix):     # 对称正定矩阵的Cholesky分解-3
    for i in range(1, A.row + 1):
        for j in range(1, i):
            for k in range(1, j):               # Note: i > j > k
                A[i ,j] -= A[i, k] * A[k, j]    # 计算t(i,j)，并存入A(i,j)中
            A[j, i] = A[i, j] / A[j, j]         # 计算l(i,j)，并存入A(j,i)中
        for k in range(1, i):
            A[i, i] -= A[i, k] * A[k, i]        # 计算d(i)，并存入A(i,i)中
    return A


def solve_1(A:Matrix, b:Matrix):
    cho(A)
    for i in range(1, A.row + 1):       # i > j
        for j in range(1, i):
            b[i] -= A[i, j] * b[j]      # A(i,j)存储的是l(i,j)
        b[i] /= A[i, i]
    for i in range(A.row, 0, -1):       # i < j
        for j in range(A.col, i, -1):
            b[i] -= A[j, i] * b[j]      # A(j,i)存储的是l(j,i)
        b[i] /= A[i, i]
    return b


def solve_2(A:Matrix, b:Matrix):
    cho2(A)
    for i in range(1, A.row + 1):
        for j in range(1, i):
            b[i] -= A[i, j] * b[j]      # A(i,j)存储的就是l(i,j)
    for i in range(A.row, 0, -1):
        b[i] /= A[i, i]
        for j in range(A.col, i, -1):
            b[i] -= A[j, i] * b[j]      # A(j,i)存储的是l(j,i)
    return b


def solve_3(A:Matrix, b:Matrix):
    cho3(A)
    for i in range(1, A.row + 1):
        for j in range(1, i):
            b[i] -= A[j, i] * b[j]      # A(j,i)存储的才是l(i,j)！
    for i in range(A.row, 0, -1):
        b[i] /= A[i, i]
        for j in range(A.col, i, -1):
            b[i] -= A[i, j] * b[j]      # A(i,j)存储的是l(j,i)
    return b


# 测试集
A, b = Matrix(3,3,0), Matrix(3,1,0)
A.matrix = [[3, 3, 5], [3, 5, 9], [5, 9, 17]]
b.matrix = [[0], [-2], [-4]]
print(A)
print(b)
print(solve_3(copy.deepcopy(A), copy.deepcopy(b)))

# 比赛开始
l1, l2, l3 = [], [], []
t1 = Timer('solve_1(copy.deepcopy(A), copy.deepcopy(b))','from __main__ import solve_1, A, b, copy')
t2 = Timer('solve_2(copy.deepcopy(A), copy.deepcopy(b))','from __main__ import solve_2, A, b, copy')
t3 = Timer('solve_3(copy.deepcopy(A), copy.deepcopy(b))','from __main__ import solve_3, A, b, copy')
l1.append(t1.timeit(number=5000))
l2.append(t2.timeit(number=5000))
l3.append(t3.timeit(number=5000))
print(l1[0])
print(l2[0])
print(l3[0])



'''
# 使用matplotlib作图(可忽略这一部分)
plt.figure()
plt.plot(range(100000,1000001,100000),l1,"-",label="insert item")
plt.plot(range(100000,1000001,100000),l2,"-",label="append item")
plt.xlabel("scale of data")
plt.ylabel("time for 1000(s)")
plt.title("Output Result")
plt.legend()
plt.show()
'''



