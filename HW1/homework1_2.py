# 计算物理第一次大作业第2题源代码
# 运行该程序大约需要0.1s

# 以下导入的均为python内置库
import copy
import time
from math import sqrt, cos, pi
import random


class Matrix:   # 矩阵类
    def __init__(self, value):
        if isinstance(value, tuple):        # 已知维度 创建矩阵
            self.size = (value[0], value[1])      # 矩阵的大小
            self.row = value[0]              # 矩阵的行数
            self.col = value[1]             # 矩阵的列数
            if len(value) > 2:
                self.matrix = [[value[2] for _ in range(value[1])] for _ in range(value[0])]     # 矩阵所有元素相同
            else:
                self.matrix = [[0 for _ in range(value[1])] for _ in range(value[0])]     # 默认创建零矩阵
        elif isinstance(value, list):       # 已知各矩阵元 创建矩阵
            self.size = (len(value), len(value[0]))      # 矩阵的大小
            self.row = len(value)                        # 矩阵的行数
            self.col = len(value[0])                     # 矩阵的列数
            self.matrix = value                          # 列表储存的就是矩阵元素

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

    def det(self):      # 矩阵的行列式
        assert self.row == self.col, "不是方阵，不能求行列式"
        n = self.row
        A = copy.deepcopy(self)
        det = 1
        for k in range(1, n):
            if A[k, k] == 0:
                for m in range(k + 1, n + 1):   # 找到绝对值最大的元素作为列主元
                    if A[m, k] != 0:
                        A[k], A[m] = A[m], A[k]     # 交换两行
                        det *= -1           # 行列式反号
            for r in range(k + 1, n + 1):   # 对k行以下每一行进行消元
                A[r, k] /= - A[k, k]     # 倍乘因子
                for c in range(k + 1, n + 1):   # 扫描该行每个元素
                    A[r, c] += A[r, k] * A[k, c]     # 更新矩阵元素
            det *= A[k, k]
        det *= A[n, n]
        return det

    def __str__(self):      # 直接输出的效果
        string = '['
        if self.row == 1 and self.col == 1:     # 只有一个矩阵元
            string += f'{self[1, 1]} ]'
        elif self.row == 1 and self.col > 1:      # 只有一行
            for c in range(1, self.col):
                string += f'{str(self[1, c])}  '
            string += f'{str(self[1, self.col])} ]'
        elif self.row > 1 and self.col == 1:      # 只有一列
            string = '[ '
            for r in range(1, self.row):
                string += f'{str(self[r, 1])}\n  '
            string += f'{str(self[self.row, 1])} ]'
        else:
            for r in range(1, self.row):
                for c in range(1, self.col):
                    if r == 1 and c == 1:
                        string += '{:^2}'.format(str(self[r, c]))
                    else:
                        string += '{:^3}'.format(str(self[r, c]))
                string += '{:^3}\n'.format(str(self[r, self.col]))
            for c in range(1, self.col):
                string += '{:^3}'.format(str(self[self.row, c]))
            string += '{:^3}]'.format(str(self[self.row, self.col]))
        return string


def cpe(A:Matrix, b:Matrix):    # 列主元消去法（column principal element）
    for k in range(1, A.row):
        index = k
        for m in range(k + 1, A.row + 1):   # 找到绝对值最大的元素作为列主元
            if abs(A[m, k]) > abs(A[index, k]):
                index = m       # 该元素的行指标记为index
        if A[index, k] == 0:
            return
        if index != k:
            A[k], A[index] = A[index], A[k]    # 交换两行
            b[k], b[index] = b[index], b[k]
        for i in range(k + 1, A.row + 1):   # 对k行以下每一行进行消元
            A[i, k] /= - A[k, k]     # 倍乘因子
            for j in range(k + 1, A.col + 1):   # 扫描该行每个元素
                A[i, j] += A[i, k] * A[k, j]     # 更新矩阵元素
            b[i] += A[i, k] * b[k]
    for i in range(A.row, 0, -1):   # 回代法解方程
        for j in range(A.col, i, -1):
            b[i] -= A[i, j] * b[j]
        b[i] /= A[i, i]
    return b      # 返回解向量

def D(m, n, x, y):      # 生成矩阵D
    D = Matrix((m*n, m*n))
    for i in range(1, m*n):
        if i % n:          # 当i为n的整数倍时，矩阵元为y或-y
            if isinstance(x, str):      # 输入为符号
                D[i, i+1] = 'x'
                D[i+1, i] = '-x'
            else:                       # 输入为数
                D[i, i+1] = x
                D[i+1, i] = -x
    for i in range(1, n + 1):
        for j in range(1, m):
            if isinstance(y, str):      # 输入为符号
                D[i + (j-1) * n, (j+1) * n - i + 1] = 'y'
                D[i + j * n, j * n - i + 1] = '-y'
            else:                       # 输入为数
                D[i + (j-1) * n, (j+1) * n - i + 1] = y
                D[i + j * n, j * n - i + 1] = -y
    return D

def dimer(m, n):    # dimer模型配分函数
    A = Matrix([[(1 + 0.1*i) ** j for j in range(int(m*n/2), -1, -1)] for i in range(int(m*n/2) + 1)])
    b = Matrix([[sqrt(D(m, n, 1 + 0.1*i, 1).det())] for i in range(int(m*n/2) + 1)])
    w = cpe(A, b)
    lw = [int(round(w[i], 1)) for i in range(1, int(m*n/2) + 2)]
    Zmn = ''
    for i in range(0, int(m*n/2) + 1):
        if lw[i] > 0:
            if i == 0:
                Zmn += f'{lw[0]}(x^{int(m*n/2)}) '
            elif i == 1:
                Zmn += f'+ {lw[i]}(x^{int(m*n/2) - i}y) '
            elif i == int(m*n/2) - 1:
                Zmn += f'+ {lw[-2]}(xy^{i}) '
            elif i == int(m*n/2):
                Zmn += f'+ {lw[-1]}(y^{int(m*n/2)}) '
            else:
                Zmn += f'+ {lw[i]}(x^{int(m*n/2) - i}y^{i}) '
        elif lw[i] < 0:
            if i == 0:
                Zmn += f'-{-lw[0]} x^{int(m*n/2)} '
            elif i == 1:
                Zmn += f'- {-lw[i]} x^{int(m*n/2) - i}y '
            elif i == int(m*n/2) - 1:
                Zmn += f'- {-lw[-2]} xy^{i} '
            elif i == int(m*n/2):
                Zmn += f'- {-lw[-1]} y^{int(m*n/2)} '
            else:
                Zmn += f'- {-lw[i]} x^{int(m*n/2) - i}y^{i} '
    Zmn = Zmn.lstrip('+')
    return Zmn if Zmn else 0

def analytical_result(m, n, x, y):      # 解析结果
    Z2 = (-1)**(m * int(n/2)) * 2**(m * n)
    for q in range(1, m+1):
        for r in range(1, n+1):
            Z2 *= complex(x * cos(pi * r / (n+1)), y * cos(pi * q / (m+1)))
    return Z2.real


time_start = time.time()


print('--------------1)3×4的正方晶格对应的D矩阵-----------------')
print(D(3, 4, 'x', 'y'))
print('')

print('-----------------2)上题中矩阵D的行列式-------------------')
print(dimer(3, 4))
print('')

print('-------------3)数值结果与解析结果的一致性验证---------------')
for i in range(1, 11):
    m, n, x, y = random.randint(1, 6), random.randint(1, 6), random.uniform(0.5, 1), random.uniform(0.5, 1)
    print(f'第{i}次比较：m = {m}, n = {n}, x = {x}, y = {y}')
    print(f'数值计算结果为：Z^2 = {round(D(m, n, x, y).det(), 10)}')
    print(f'解析计算结果为：Z^2 = {round(analytical_result(m, n, x, y), 10)}')      # 都最多保留10位小数
print('')

print('----------4)m、n都为奇数时dimer模型配分函数为零-------------')
print(f'3 × 3 正方晶格的配分函数为：{dimer(3, 3)}')
print('')


time_end = time.time()
print(f'运行结束，耗时{time_end - time_start}s')

