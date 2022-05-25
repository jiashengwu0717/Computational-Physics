import math
import copy

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



# 线性方程组的求解


def utm(U:Matrix, b:Matrix, U0 = False):     # 上三角矩阵的回代算法(upper triangle matrix)
    x = copy.deepcopy(b)    # 不改变向量b
    for i in range(U.row, 0, -1):
        assert U[i, i] != 0, '方程无解！'
        for j in range(U.col, i, -1):
            x[i] -= U[i, j] * x[j]
        if not U0:      # 对单位上三角矩阵，U[i, i] = 1
            x[i] /= U[i, i]
    return x    # 返回解向量

print('--------上三角矩阵解方程测试--------')
A, b = Matrix((3,3)), Matrix((3,1))
A.matrix = [[1, -1, 4], [0, 2, 3], [0, 0, 1]]
b.matrix = [[1], [7], [1]]
x = utm(A, b)
print(x)
print(b)


def ltm(L:Matrix, b, L0 = False):      # 下三角矩阵的前代算法(lower triangle matrix)
    x = copy.deepcopy(b)
    for i in range(1, L.row + 1):
        assert L[i, i] != 0, '方程无解！'
        for j in range(1, i):
            x[i] -= L[i, j] * x[j]
        if not L0:      # 对单位下三角矩阵，L[i, i] = 1
            x[i] /= L[i, i]
    return x

print('--------下三角矩阵解方程测试---------')
A, b = Matrix((3,3)), Matrix((3,1))
A.matrix = [[1, 0, 0], [3, 2, 0], [4, -1, 1]]
b.matrix = [[1], [7], [1]]
x = ltm(A, b)
print(x)
print(b)


def gauss(A:Matrix, b:Matrix):    # 高斯消元法
    for k in range(1, A.row):     # k为某一基准行
        assert A[k, k] != 0, '消不出来'
        for i in range(k + 1, A.row + 1):   # 对k行以下每一行进行消元
            c = - A[i, k] / A[k, k]     # 倍乘因子
            for j in range(k + 1, A.col + 1):   # 扫描该行每个元素
                A[i, j] += c * A[k, j]     # 更新矩阵元素
            b[i] += c * b[k]
    return A, b     # 此时的矩阵A没有将消去的值归零，但可直接作为上三角矩阵进行运算

print('----------高斯消元法测试-----------')
A, b = Matrix((3,3)), Matrix((3,1))
A.matrix = [[2, 1, 1], [1, 3, 2], [1, 2, 2]]
b.matrix = [[4], [6], [5]]
A, b = gauss(A, b)
print(A)
print(b)
x = utm(A, b)
print(x)


def lud(A:Matrix):      # 矩阵的LU分解 (LU-decomposition)
    for k in range(1, A.row):
        if A[k ,k] == 0:
            break
        for i in range(k + 1, A.row + 1):   # 计算L的第k列
            for j in range(1, k):
                A[i, k] -= A[j, k] * A[i, j]
            A[i, k] /= A[k, k]
        for j in range(k + 1, A.col + 1):   # 计算U的第k+1列
            for i in range(1, k + 1):
                A[k+1, j] -= A[k+1, i] * A[i, j]
    return A

print('-----------LU分解测试-------------')
A, b = Matrix((3,3)), Matrix((3,1))
A.matrix = [[1, 2, 3], [2, 5, 2], [3, 1, 5]]
b.matrix = [[14], [18], [20]]
A = lud(A)
print(A)
y = ltm(A, b, True)
x = utm(A, y)
print(x)


def normal_solve(A:Matrix, b:Matrix):   # 一阶线性方程组的一般解法
    A = lud(A)
    y = ltm(A, b, True)
    x = utm(A, y)
    return x


def tdm(a:list, b:list, c:list, f:list):      # triple diagonal matrix 追赶法
    x, n = [], len(b)                          # 特别注意此处各变量均为list类型！
    for i in range(1, n):
        x.append(a[i - 1] / b[i - 1])   # 此时列表x[i]中储存的是alpha[i]
        b[i] -= x[-1] * c[i - 1]        # 将计算出的l[i]存入b[i]中
        f[i] -= x[-1] * f[i - 1]        # L0^{-1}左乘右端列向量f的结果
    x.append(f[-1] / b[-1])             # 从该行起方程组的解x[i]覆盖掉alpha[i]
    for i in range(n-2, -1, -1):
        x[i] = (f[i] - c[i] * x[i+1]) / b[i]    # 回代法解上三角系数矩阵方程组
    return x

print('--------三对角矩阵的追赶法-------')
a,b,c,f = [3, -2], [1, 2, -1] , [-4, 5], [1, 8 ,-1]
print(tdm(a, b, c, f))      # x为list类型


def cpe(A:Matrix, b:Matrix = None):    # 列主元消去法（column principal element）
    if not b:
        b = Matrix((A.row, 1))    # 这样不输入b也能计算A的行列式
    det = 1
    for k in range(1, A.row):
        index = k
        for m in range(k + 1, A.row + 1):   # 找到绝对值最大的元素作为列主元
            if abs(A[m, k]) > abs(A[index, k]):
                index = m   # 该元素的行指标记为index
        if A[index, k] == 0:
            return 0    # 返回行列式为0
        if index != k:
            A[k], A[index] = A[index], A[k]     # 交换两行
            b[k], b[index] = b[index], b[k]
            det *= -1
        for i in range(k + 1, A.row + 1):   # 对k行以下每一行进行消元
            A[i, k] /= - A[k, k]     # 倍乘因子
            for j in range(k + 1, A.col + 1):   # 扫描该行每个元素
                A[i, j] += A[i, k] * A[k, j]     # 更新矩阵元素
            b[i] += A[i, k] * b[k]
        det *= A[k, k]
    det *= A[A.row, A.col]
    for i in range(A.row, 0, -1):   # 回代法解方程
        for j in range(A.col, i, -1):
            b[i] -= A[i, j] * b[j]
        b[i] /= A[i, i]
    return det, b      # 返回矩阵行列式的值和解向量

print('---------列主元消去法求矩阵的行列式---------')
A = Matrix((3,3))
A.matrix = [[12, -3, 3], [-18, 3, -1], [1, 1, 1]]
print(A)
det = cpe(A)[0]
print(f'行列式为：{det}')

print('----------列主元消去法解方程-----------')
A, b = Matrix((3,3)), Matrix((3,1))
A.matrix = [[0.5, 1.1, 3.1], [2, 4.5, 0.36], [5, 0.96, 6.5]]
b.matrix = [[6], [0.02], [0.96]]
x = cpe(A, b)[1]
print(x)


def gje(A:Matrix, b:Matrix):     # Gauss-Jordan Elimination(列主元高斯-约当消去法)
    for k in range(1, A.row + 1):
        index = k
        for m in range(k + 1, A.row + 1):   # 找到绝对值最大的元素作为列主元
            if abs(A[m, k]) > abs(A[index, k]):
                index = m   # 该元素的行指标记为index
        if A[index, k] == 0:
            break       # 别解了
        if index != k:
            A.exchange(k, index)    # 交行两行
            b.exchange(k, index)
        for i in range(1, A.row + 1):   # 对第k行外的每一行进行消元
            if i != k:
                A[i, k] /= - A[k, k]
                for j in range(k + 1, A.col + 1):
                    A[i, j] += A[i, k] * A[k, j]
                b[i] += A[i, k] * b[k]
        for j in range(k + 1, A.col + 1):    # 主行主元归一
            A[k, j] /= A[k, k]
        b[k] /= A[k, k]         # 循环和空格！
    return b

print('----------高斯-约当消元法-----------')
A, b = Matrix((3,3)), Matrix((3,1))
A.matrix = [[4, 3, 1], [2, 1, 2], [6, 1, 5]]
b.matrix = [[11], [6], [13]]
print(A)
x = gje(A, b)
print(x)


def inverse(A:Matrix):      # Gauss-Jordan Inverse 高斯-约当消元法求逆矩阵
    assert A.row == A.col, "行数不等于列数，没有逆矩阵"
    I = Matrix((A.row, A.col))
    for i in range(A.row):
        I[i, i] = 1         # 单位矩阵
    for k in range(1, A.row + 1):
        index = k
        for m in range(k + 1, A.row + 1):   # 找到绝对值最大的元素作为列主元
            if abs(A[m, k]) > abs(A[index, k]):
                index = m   # 该元素的行指标记为index
        if A[index, k] == 0:
            break       # 别求了
        if index != k:
            A.exchange(k, index)    # 交换两行
            I.exchange(k, index)
        for i in range(1, A.row + 1):   # 对第k行外的每一行进行消元
            if i != k:
                A[i, k] /= - A[k, k]
                for j in range(k + 1, A.col + 1):
                    A[i, j] += A[i, k] * A[k, j]
                for j in range(1, I.col + 1):
                    I[i, j] += A[i, k] * I[k, j]
        for j in range(k + 1, A.col + 1):    # 主行主元归一
            A[k, j] /= A[k, k]
        for j in range(1, I.col + 1):
            I[k, j] /= A[k, k]         # 循环和空格！
    return I

print('----------高斯-约当消元求逆矩阵-----------')
A = Matrix((3,3))
A.matrix = [[1, 2, 3], [2, 4, 5], [3, 5, 6]]
print(inverse(A))


def cplu(A:Matrix):        # column principal LU decomposition 列主元消去法进行LU分解
    p = [_ for _ in range(1, A.row + 1)]   # 列表p存储交换行的信息
    for k in range(1, A.row):
        index = k
        for m in range(k + 1, A.row + 1):   # 找到绝对值最大的元素作为列主元
            if abs(A[m, k]) > abs(A[index, k]):
                index = m   # 该元素的行指标记为index
        if A[index, k] == 0:
            break       # 别分解了
        if index != k:
            A[k], A[index] = A[index], A[k]    # 交换两行
            s = p[k - 1]
            p[k - 1] = p[index - 1]
            p[index - 1] = s            # 交换p中的两个元素位置，注意列表的索引从0开始！
        for i in range(k + 1, A.row + 1):   # 对k行以下每一行进行消元
            A[i, k] /= A[k, k]     # 倍乘因子，注意写入A(i,k)的不能带负号！否则无法得到正确的L阵
            for j in range(k + 1, A.col + 1):   # 扫描该行每个元素
                A[i, j] -= A[i, k] * A[k, j]     # 更新矩阵元素
    return A, p

print('--------列主元高斯消去法进行LU分解--------')
A = Matrix((3,3))
A.matrix = [[1, 2, 2], [4, 4, 2], [4, 6, 4]]
x = cplu(A)
print(x[0])
print(x[1])


def cholesky(A:Matrix):         # 对称正定矩阵的Cholesky分解
    for j in range(1, A.col + 1):
        for k in range(1, j):
            A[j, j] -= A[j, k] ** 2
        A[j, j] = math.sqrt(A[j, j])
        for i in range(j + 1, A.row + 1):
            for k in range(1, j):
                A[i, j] -= A[i, k] * A[j, k]
            A[i, j] /= A[j, j]
    return A

def irm2(A:Matrix):      # improved root method 2.0
    for i in range(1, A.row + 1):
        for j in range(1, i):
            for k in range(1, j):
                A[i, j] -= A[i, k] * A[k, k] * A[j, k]
            A[i, j] /= A[j, j]
        for k in range(1, i):
            A[i, i] -= A[i ,k] ** 2 * A[k, k]
    return A

def irm3(A:Matrix):      # improved root method 3.0
    for i in range(1, A.row + 1):
        for j in range(1, i):
            for k in range(1, j):               # Note: i > j > k
                A[i ,j] -= A[i, k] * A[k, j]    # 计算t(i,j)，并存入A(i,j)中
            A[j, i] = A[i, j] / A[j, j]         # 计算l(i,j)，并存入A(j,i)中
        for k in range(1, i):
            A[i, i] -= A[i, k] * A[k, i]        # 计算d(i)，并存入A(i,i)中
    return A

print('---------对称正定矩阵平方根法---------')
A, b = Matrix((3,3)), Matrix((3,1))
A.matrix = [[3, 3, 5], [3, 5, 9], [5, 9, 17]]
b.matrix = [[0], [-2], [-4]]
print(A)
cholesky(A)
print(A)


def spb(C:Matrix, b:Matrix):        # Symmetric positive definite band coefficient equations
    n, m = C.row, C.col - 1
    Lt = Matrix((n, m + 1))         # Lt记录了矩阵L的元素
    bt = Matrix((n, 1))
    for i in range(1, n + 1):
        r = 1 if (i <= m + 1) else i - m      # 只需从第r列开始计算
        for j in range(r, i + 1):
            Lt[i, j-i+m+1] = C[i, j-i+m+1] - sum([Lt[i, k-i+m+1] * Lt[j, k-j+m+1] / Lt[k, m+1] for k in range(r, j)])
            bt[i] = b[i] - sum([Lt[i, j-i+m+1] * bt[j] / Lt[j, m+1] for j in range(r, i)])
    # 分解完成，下面开始解方程，并把解写入b中
    for i in range(n, 0, -1):
        t = n if (i > n - m - 1) else i + m     # 只需计算到第t列
        b[i] = (bt[i] - sum([Lt[j, i-j+m+1] * b[j] for j in range(i+1, t+1)])) / Lt[i, m+1]
    return b

print('--------对称正定带状系数矩阵方程组---------')
C = Matrix([[0, 3], [-1, 5], [2, 4], [-1, 1]])      # 矩阵C的存储方法要留意！
b = Matrix([[2], [4], [0], [2]])
x = spb(C, b)
print(x)


def jacobi(A:Matrix,b:Matrix,M:int,e = 0,x0 = None):      # Jacobi 迭代法
    if not x0:
        x0 = Matrix((A.row, 1))      # 默认从0向量开始迭代
    x = Matrix((A.row, 1))
    k = 0
    while k < M:    # 最多迭代M次
        for i in range(1, A.row + 1):
            x[i] = (b[i] - sum([A[i, j] * x0[j] for j in range(1, A.col+1) if j != i])) / A[i, i]
        if (x - x0).norm() < e:     # 默认列和范数
            break
        x0 = copy.deepcopy(x)       # 这样赋值必须deepcopy
        k += 1
    return x

print('---------Jacobi迭代法解线性方程组--------')
A = Matrix([[10, -2, -1], [-2, 10, -1], [-1, -2, 5]])
b = Matrix([[3], [15], [10]])
print(A)
for M in range(1, 6):
    print(f'第{M}次迭代结果为{jacobi(A, b, M)}')


def gauss_seidel(A:Matrix, b:Matrix, M:int, e = 0, x0 = None):     # Gauss-Seidel 迭代法
    if not x0:
        x0 = Matrix((A.row, 1))      # 默认从0向量开始迭代
    x = Matrix((A.row, 1))
    k = 0
    while k < M:    # 最多迭代M次
        x[1] = (b[1] - sum([A[1, j] * x0[j] for j in range(2, A.col+1)])) / A[1, 1]
        for i in range(2, A.row):
            x[i] = (b[i] - sum([A[i, j] * x[j] for j in range(1, i)]) -
                    sum([A[i, j] * x0[j] for j in range(i + 1, A.col + 1)])) / A[i, i]
        x[A.row] = (b[A.row] - sum([A[A.row, j] * x[j] for j in range(1, A.col)])) / A[A.row, A.col]
        if (x - x0).norm() < e:     # 默认列和范数
            break
        x0 = copy.deepcopy(x)       # 这样赋值必须deepcopy
        k += 1
    return x

print('-------Gauss-seidel迭代法解线性方程组-------')
A = Matrix([[10, -1, -2], [-1, 10, -2], [-1, -1, 5]])
b = Matrix([[72], [83], [42]])
print(A)
for M in range(1, 6):
    print(f'第{M}次迭代结果为{gauss_seidel(A, b, M)}')


def sor(A:Matrix, b:Matrix, M:int, w=1.0, e=0, x0=None):      # successive over relaxation method
    if not x0:
        x0 = Matrix((A.row, 1))      # 默认从0向量开始迭代
    x = Matrix((A.row, 1))
    k = 0
    while k < M:    # 最多迭代M次
        x[1] = (1 - w) * x0[1] + w * (b[1] - sum([A[1, j] * x0[j] for j in range(2, A.col+1)])) / A[1, 1]
        for i in range(2, A.row):
            x[i] = (1 - w) * x0[i] + w * (b[i] - sum([A[i, j] * x[j] for j in range(1, i)]) -
                    sum([A[i, j] * x0[j] for j in range(i + 1, A.col + 1)])) / A[i, i]
        x[A.row] = (1 - w) * x0[A.row] + w * (b[A.row] - sum([A[A.row, j] * x[j] for j in range(1, A.col)])) / A[A.row, A.col]
        if (x - x0).norm() < e:     # 默认列和范数
            break
        x0 = copy.deepcopy(x)       # 这样赋值必须deepcopy
        k += 1
    return x

print('-------逐次超松弛法解线性方程组-------')
A = Matrix([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
b = Matrix([[1], [0], [1.8]])
x0 = Matrix([[1], [1], [1]])
print(A)
for M in range(1, 10):
    print(f'第{M}次迭代结果为{sor(A, b, M, 1.4, 0, x0)}')


def sdm(A:Matrix, b:Matrix, x:Matrix, M=10):      # steepest descent method 默认下降10次
    r = b - A * x
    k = 0
    while k < M:
        alpha = sum([r[i] ** 2 for i in range(r.row)]) \
                / sum([r[i] * A[i, j] * r[j] for i in range(A.row) for j in range(A.col)])
        x += r * alpha      # 数乘矩阵时数只能写在矩阵后面
        r -= A * r * alpha
        k += 1
    return x        # 此时最初输入的x不变！

print('---------最速下降法解方程---------')
A = Matrix([[3, 2], [2, 6]])
b = Matrix([[2], [-8]])
x0 = Matrix([[-2], [-2]])
print(sdm(A, b, x0, 10))
print(x0)


def cgm(A:Matrix, b:Matrix, x:Matrix, M = 10):     # conjugate gradient Method 默认下降10次
    r = b - A * x
    p = r
    k = 0
    while k < M and p.norm():        # 若p的范数为0，即p为零向量，终止下降
        alpha = sum([r[i] ** 2 for i in range(r.row)]) \
                / sum([p[i] * A[i, j] * p[j] for i in range(A.row) for j in range(A.col)])
        x += p * alpha
        rt = r
        r -= A * p * alpha
        beta = sum([r[i] ** 2 for i in range(r.row)]) / sum([rt[i] ** 2 for i in range(rt.row)])
        p = r + p * beta
        k += 1
    return x        # 此时最初输入的x不变！

print('---------共轭梯度法解方程---------')
A = Matrix([[3, 2], [2, 6]])
b = Matrix([[2], [-8]])
x0 = Matrix([[-2], [-2]])
print(cgm(A, b, x0))
print(x0)


# 线性方程组解法 完结撒花！







