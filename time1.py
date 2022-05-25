import copy
from math import sqrt, sin, cos
from timeit import Timer


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

    def __eq__(self, other):    # 判断维度是否相等
        assert isinstance(other, Matrix), "类型不匹配，不能比较"
        return other.size == self.size

    def __add__(self, N):   # 加减乘运算中，r、c的含义无需从1开始取，只要保证遍历整个矩阵即可
        assert N.size == self.size, "维度不匹配，不能相加"
        M = Matrix((self.row, self.col))
        for r in range(self.row):
            for c in range(self.col):
                M[r, c] = self[r, c] + N[r, c]
        return M

    def __sub__(self, N):
        assert N.size == self.size, "维度不匹配，不能相减"
        M = Matrix((self.row, self.col))
        for r in range(self.row):
            for c in range(self.col):
                M[r, c] = self[r, c] - N[r, c]
        return M

    def __mul__(self, N):
        if isinstance(N, int) or isinstance(N, float):  # 数乘
            M = Matrix((self.row, self.col))
            for r in range(self.row):
                for c in range(self.col):
                    M[r, c] = self[r, c] * N
        else:     # 矩阵乘法
            assert N.row == self.col, "维度不匹配，不能相乘"
            M = Matrix((self.row, N.col))
            for r in range(self.row):
                for c in range(N.col):
                    sum = 0
                    for i in range(self.col):
                        sum += self[r, i] * N[i, c]
                    M[r, c] = sum
        return M

    def exchange(self, i, j):   # 交换矩阵的i行和j行
        r = self[i]
        self[i] = self[j]
        self[j] = r

    def __reversed__(self):     # 矩阵的转置（原矩阵不变）
        X = Matrix((self.col, self.row))
        for r in range(self.row):
            for c in range(self.col):
                X[c, r] = self[r, c]
        return X

    def norm(self, p=1):        # 范数，默认求列和范数
        if p == 1:          # 1-范数;列和范数
            norm = sum([abs(self.matrix[r][0]) for r in range(self.row)])
            for c in range(1, self.col):
                s = sum([abs(self.matrix[r][c]) for r in range(self.row)])
                if s > norm:
                    norm = s
        elif p == 0:        # inf-范数；行和范数
            norm = sum([abs(self.matrix[0][c]) for c in range(self.col)])
            for r in range(1, self.row):
                s = sum([abs(self.matrix[r][c]) for c in range(self.col)])
                if s > norm:
                    norm = s
        else:
            norm = 0
        return norm

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

def cgm(A:Matrix, b:Matrix, x:Matrix, M = 5):      # conjugate gradient Method
    r = b - A * x
    p = r
    k = 0
    while k < M:
        alpha = sum([r[i] * p[i] for i in range(r.row)]) \
                / sum([r[i] * A[i, j] * p[j] for i in range(A.row) for j in range(A.col)])
        x += p * alpha
        r -= A * p * alpha
        beta = - sum([r[i] * A[i, j] * p[j] for i in range(A.row) for j in range(A.col)]) \
               / sum([p[i] * A[i, j] * p[j] for i in range(A.row) for j in range(A.col)])
        p = r + p * beta
        k += 1
    return x        # 此时最初输入的x不变！

def cgm2(A:Matrix, b:Matrix, x:Matrix, M = 5):
    r = b - A * x
    p = r
    k = 0
    while k < M:
        alpha = sum([r[i] ** 2 for i in range(r.row)]) \
                / sum([p[i] * A[i, j] * p[j] for i in range(A.row) for j in range(A.col)])
        x += p * alpha
        rt = r
        r -= A * p * alpha
        beta = sum([r[i] ** 2 for i in range(r.row)]) / sum([rt[i] ** 2 for i in range(rt.row)])
        p = r + p * beta
        k += 1
    return x


def tdm(a:list, b:list, c:list, f:list):      # triple diagonal matrix 追赶法
    n = len(b)                              # 特别注意此处各变量均为list类型！
    for i in range(1, n):
        m = a[i - 1] / b[i - 1]
        b[i] -= m * c[i - 1]
        f[i] -= m * f[i - 1]
    x = [f[-1] / b[-1]]
    for i in range(n-2, -1, -1):
        x.insert(0, (f[i] - c[i] * x[0]) / b[i])
    return x

def tdm2(a:list, b:list, c:list, f:list):      # triple diagonal matrix 追赶法
    x, n = [], len(b)                          # 特别注意此处各变量均为list类型！
    for i in range(1, n):
        x.append(a[i - 1] / b[i - 1])   # 此时列表x[i]中储存的是alpha[i]
        b[i] -= x[-1] * c[i - 1]        # 将计算出的l[i]存入b[i]中
        f[i] -= x[-1] * f[i - 1]        # L0^{-1}左乘右端列向量f的结果
    x.append(f[-1] / b[-1])             # 从该行起方程组的解x[i]覆盖掉alpha[i]
    for i in range(n-2, -1, -1):
        x[i] = (f[i] - c[i] * x[i+1]) / b[i]
    return x


'''
# 共轭梯度下降法测速实验
A = Matrix([[3, 2], [2, 6]])
b = Matrix([[2], [-8]])
x0 = Matrix([[-2], [-2]])

t1 = Timer('cgm(copy.deepcopy(A), copy.deepcopy(b), copy.deepcopy(x0))','from __main__ import cgm, A, b, x0, copy')
t2 = Timer('cgm2(copy.deepcopy(A), copy.deepcopy(b), copy.deepcopy(x0))','from __main__ import cgm2, A, b, x0, copy')
print(f'2000次实验所用时间为：{t1.timeit(number=2000)}')
print(f'改进后2000次实验所用时间为：{t2.timeit(number=2000)}')
'''

'''
# 三对角矩阵追赶法测速实验
a,b,c,f = [3, -2], [1, 2, -1] , [-4, 5], [1, 8 ,-1]
t1 = Timer('tdm(copy.deepcopy(a), copy.deepcopy(b), copy.deepcopy(c), copy.deepcopy(f))','from __main__ import tdm, a, b, c, f ,copy')
t2 = Timer('tdm2(copy.deepcopy(a), copy.deepcopy(b), copy.deepcopy(c), copy.deepcopy(f))','from __main__ import tdm2, a, b, c, f, copy')
print(f'5000次实验所用时间为：{t1.timeit(number=5000)}')
print(f'改进后5000次实验所用时间为：{t2.timeit(number=5000)}')
'''






