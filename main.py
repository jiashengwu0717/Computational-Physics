# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/




import itertools
from copy import deepcopy
from math import sqrt

# 引用该矩阵类时索引从1开始取，即矩阵A的(i, j)元直接用A[i, j]取用

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
                self.matrix[key - 1] = deepcopy(value)   # 设置矩阵的第i行
            elif self.col == 1:
                self.matrix[key - 1][0] = value     # 设置列向量的第i个分量
        elif isinstance(key, tuple):
            self.matrix[key[0] - 1][key[1] - 1] = value

    def getRow(self, item):         # 取出某一行的行向量
        return Matrix([self[item]])

    def getCol(self, item):         # 取出某一列的列向量
        return Matrix([[self[r, item]] for r in range(1, self.row + 1)])

    def getRow_list(self, item):    # 取出某一行元素
        return [x for x in self.getRow(item)]

    def getCol_list(self, item):    # 取出某一列元素
        return [x for x in self.getCol(item)]

    def getSubMatrix(self, r:tuple, c:tuple):   # 取出子矩阵
        r1, r2 = r
        c1, c2 = c
        return Matrix([[self[i, j] for j in range(c1, c2+1)] for i in range(r1, r2+1)])

    def reverse(self):     # 矩阵的转置（原矩阵不变）
        X = Matrix((self.col, self.row))
        for r, c in itertools.product(range(self.row), range(self.col)):
            X[c, r] = self[r, c]
        return X

    def inverse(self):  # 矩阵的逆
        assert self.row == self.col, '行数不等于列数，不能求逆'
        n = self.row
        A = deepcopy(self)     # 求逆矩阵结束，不改变原矩阵
        I = Matrix([[1 if i == j else 0 for i in range(n)] for j in range(n)])  # 单位矩阵
        for k in range(1, n + 1):
            index = k
            for m in range(k + 1, n + 1):   # 找到绝对值最大的元素作为列主元
                if abs(A[m, k]) > abs(A[index, k]):
                    index = m   # 该元素的行指标记为index
            if A[index, k] == 0:
                return          # 逆矩阵不存在，返回None
            if index != k:
                A[k], A[index] = A[index], A[k]     # 交换两行
                I[k], I[index] = I[index], I[k]
            for i in range(1, n + 1):       # 对第k行外的每一行进行消元
                if i != k:
                    A[i, k] /= - A[k, k]
                    for j in range(k + 1, n + 1):
                        A[i, j] += A[i, k] * A[k, j]
                    for j in range(1, n + 1):
                        I[i, j] += A[i, k] * I[k, j]
            for j in range(k + 1, n + 1):    # 主行主元归一
                A[k, j] /= A[k, k]
            for j in range(1, n + 1):
                I[k, j] /= A[k, k]
        return I

    def __eq__(self, other):    # 判断维度是否相等
        assert isinstance(other, Matrix), '类型不匹配，不能比较'
        return other.size == self.size and [x for x in other] == [x for x in self]

    def __add__(self, N):   # 加减乘运算中，r、c的含义无需从1开始取，只要保证遍历整个矩阵即可
        assert N.size == self.size, '维度不匹配，不能相加'
        M = Matrix((self.row, self.col))
        for r, c in itertools.product(range(self.row), range(self.col)):
            M[r, c] = self[r, c] + N[r, c]
        return M

    def __sub__(self, N):
        assert N.size == self.size, '维度不匹配，不能相减'
        M = Matrix((self.row, self.col))
        for r, c in itertools.product(range(self.row), range(self.col)):
            M[r, c] = self[r, c] - N[r, c]
        return M

    def __mul__(self, N):
        if isinstance(N, int) or isinstance(N, float):  # 数乘，且数只能写在右边！
            M = Matrix((self.row, self.col))
            for r, c in itertools.product(range(self.row), range(self.col)):
                M[r, c] = self[r, c] * N
        else:     # 矩阵乘法
            assert N.row == self.col, '维度不匹配，不能相乘'
            import numpy as np      # 使用numpy矩阵乘法加速一下
            M = np.matmul(np.matrix(self.matrix), np.matrix(N.matrix))
            M = Matrix([[M[r].flat[c] for c in range(N.col)] for r in range(self.row)])
        return M

    '''
    def __mul__(self, N):
        if isinstance(N, int) or isinstance(N, float):  # 数乘，且数只能写在右边！
            M = Matrix((self.row, self.col))
            for r, c in itertools.product(range(self.row), range(self.col)):
                M[r, c] = self[r, c] * N
        else:     # 矩阵乘法
            assert N.row == self.col, '维度不匹配，不能相乘'
            M = Matrix((self.row, N.col))
            for r, c in itertools.product(range(self.row), range(N.col)):
                sum = 0
                for i in range(self.col):
                    sum += self[r, i] * N[i, c]
                M[r, c] = sum
        return M
    '''

    def __divmod__(self, N):
        pass

    def __pow__(self, power, modulo=None):
        assert self.row == self.col, '不是方阵，不能乘方'
        M = deepcopy(self)
        for i in range(power):
            M = M * self
        return M

    def __str__(self, preference = 2, width = 5):          # 直接输出的效果
        string = '['
        if self.row == 1 and self.col == 1:     # 只有一个矩阵元
            string += f'{round(self[1, 1], preference)} ]'
        elif self.row == 1 and self.col > 1:      # 只有一行
            for c in range(1, self.col):
                string += f'{str(round(self[1, c], preference))}  '
            string += f'{str(round(self[1, self.col], preference))} ]'
        elif self.row > 1 and self.col == 1:      # 只有一列
            string = '[ '
            for r in range(1, self.row):
                string += f'{str(round(self[r, 1], preference))}\n  '
            string += f'{str(round(self[self.row, 1], preference))} ]'
        else:
            for r in range(1, self.row):
                for c in range(1, self.col):
                    if r == 1 and c == 1:
                        string += f'{round(self[r, c], preference):^{width - 1}}'
                    else:
                        string += f'{round(self[r, c], preference):^{width}}'
                string += f'{round(self[r, self.col], preference):^{width}}\n'
            for c in range(1, self.col):
                string += f'{round(self[self.row, c], preference):^{width}}'
            string += f'{round(self[self.row, self.col], preference):^{width - 1}}]'
        return string

    def det(self):      # 矩阵的行列式
        assert self.row == self.col, '不是方阵，不能求行列式'
        n = self.row
        A = deepcopy(self)     # 求行列式结束，不改变原矩阵
        det = 1
        for k in range(1, n):
            index = k
            for m in range(k + 1, n + 1):   # 找到绝对值最大的元素作为列主元
                if abs(A[m, k]) > abs(A[index, k]):
                    index = m       # 该元素的行指标记为index
            if A[index, k] == 0:
                return 0    # 返回行列式为0
            if index != k:
                A[k], A[index] = A[index], A[k]    # 交换两行
                det *= -1           # 行列式反号
            for r in range(k + 1, n + 1):   # 对k行以下每一行进行消元
                A[r, k] /= - A[k, k]     # 倍乘因子
                for c in range(k + 1, n + 1):   # 扫描该行每个元素
                    A[r, c] += A[r, k] * A[k, c]     # 更新矩阵元素
            det *= A[k, k]
        det *= A[n, n]
        return det

    def tr(self):       # 矩阵的迹
        assert self.row == self.col, '不是方阵，不能计算迹'
        return sum([self[r, r] for r in range(self.row)])

    def exchange(self, i, j):   # 交换矩阵的i行和j行
        r = self[i]
        self[i] = self[j]
        self[j] = r

    def exchange_col(self, i, j):   # 交换矩阵的i列和j列
        r = [self[k, i] for k in range(self.row)]
        for k in range(self.row):
            self[k, i] = self[k, j]
        for k in range(self.row):
            self[k, j] = r[k]

    def norm(self, p=1):        # 范数，默认求列和范数
        if p == 1:          # 1-范数;列和范数
            norm = sum([abs(self.matrix[r][0]) for r in range(self.row)])
            for c in range(1, self.col):
                s = sum([abs(self.matrix[r][c]) for r in range(self.row)])
                if s > norm:
                    norm = s
        elif p == 0:         # inf-范数；行和范数
            norm = sum([abs(self.matrix[0][c]) for c in range(self.col)])
            for r in range(1, self.row):
                s = sum([abs(self.matrix[r][c]) for c in range(self.col)])
                if s > norm:
                    norm = s
        elif p == 2:         # 2-范数；谱范数
            if self.col == 1:
                norm = sqrt(sum(x**2 for x in self))
            else:
                D = self.reverse() * self
                norm = sqrt(max(D.eigenvalue()))
        else:
            norm = 0
        return norm

    def cond(self, p):     # 条件数
        return self.norm(p) * self.inverse().norm(p)

    def GHR(self):  # Givens Hessenberg Reduction——QR
        n = self.row
        A = deepcopy(self)
        G = I_Matrix(n)     # 变换矩阵
        for j in range(1, n):
            for i in range(n, j, -1):
                if A[i, j]:
                    c, s = A[i - 1, j] / sqrt(A[i - 1, j] ** 2 + A[i, j] ** 2), A[i, j] / sqrt(
                        A[i - 1, j] ** 2 + A[i, j] ** 2)
                    g = Jacobi_Matrix(n, i - 1, i, c, s)  # 记录每一步变换
                    G = G * g.reverse()
                    A = g * A
        return G, A     # Q、R

    def eigenvalue(self):      # 本征值（QR迭代算法求解实特征值）
        over = lambda A: max([abs(A[i, j]) for i in range(1, A.row) for j in range(i+1, A.col+1)])
        A = deepcopy(self)
        # while over(R) > 10**-2:
        for _ in range(100):
            Q, R = A.GHR()
            A = R * Q
        return [round(A[i, i], 10) for i in range(1, A.row + 1)]       # 返回特征值列表

    def __iter__(self):     # 遍历所有元素
        return iter(self[r, c] for r in range(1, self.row+1) for c in range(1, self.col+1))

class Vector:                           # 向量类
    def __init__(self, value):
        if isinstance(value, tuple):    # 已知维度 创建向量
            self.size = value[0]
            if len(value) > 1:
                self.vec = [value[1] for _ in range(value[0])]
            else:
                self.vec = [0 for _ in range(value[0])]
        elif isinstance(value, list):   # 已知各分量 创建向量
            self.size = len(value)
            self.vec = value

    def __getitem__(self, item):            # 按索引取值
        return self.vec[item - 1]           # 索引从1开始

    def __setitem__(self, key, value):      # 给某分量赋值
        self.vec[key - 1] = value

    def __add__(self, v):           # 加法运算
        assert v.size == self.size, '维度不匹配，不能相加'
        return Vector([self[i] + v[i] for i in range(1, self.size+1)])

    def __sub__(self, v):           # 减法运算
        assert v.size == self.size, '维度不匹配，不能相减'
        return Vector([self[i] - v[i] for i in range(1, self.size+1)])

    def __mul__(self, t):           # 数乘
        return Vector([x * t for x in self.vec])

    def __truediv__(self, t):       # 数除
        return Vector([x / t for x in self.vec])

    def __str__(self):              # 输出效果：以元组形式打印各分量
        return str(tuple(self.vec))



from math import sqrt
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Songti SC']
import numpy as np
from scipy.fftpack import fft, ifft

