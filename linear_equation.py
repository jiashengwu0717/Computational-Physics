import math
import copy
from timeit import Timer
from matrix import Matrix


# 对于对称正定矩阵

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


def sspdm(A:Matrix, b:Matrix, x=None, M=10, band = False):        # solve symmetric positive definite matrix
    if not x:
        if band:        # 带状矩阵可少储存、少搜索某些元素
            return spb(A, b)
        # 一般使用cholesky分解法直接求解
        for i in range(1, A.row + 1):
            for j in range(1, i):
                for k in range(1, j):               # Note: i > j > k
                    A[i ,j] -= A[i, k] * A[k, j]    # 计算t(i,j)，并存入A(i,j)中
                A[j, i] = A[i, j] / A[j, j]         # 计算l(i,j)，并存入A(j,i)中
            for k in range(1, i):
                A[i, i] -= A[i, k] * A[k, i]        # 计算d(i)，并存入A(i,i)中
        # cholesky分解完成
        for i in range(1, A.row + 1):
            for j in range(1, i):
                b[i] -= A[j, i] * b[j]      # A(j,i)存储的才是l(i,j)！
        for i in range(A.row, 0, -1):
            b[i] /= A[i, i]
            for j in range(A.col, i, -1):
                b[i] -= A[i, j] * b[j]      # A(i,j)存储的是l(j,i)
        return b
    else:       # 此时从x开始使用共轭梯度法求解，默认下降10次
        r = b - A * x
        p = r
        k = 0
        while k < M and p.norm():      # p的范数为0，即p为零向量，终止下降
            alpha = sum(r[i]**2 for i in range(r.row)) / sum(p[i] * A[i, j] * p[j] for i in range(A.row) for j in range(A.col))
            x += p * alpha
            rt = r
            r -= A * p * alpha
            beta = sum(r[i] ** 2 for i in range(r.row)) / sum(rt[i] ** 2 for i in range(rt.row))
            p = r + p * beta
            k += 1
        return x


A = Matrix([[4, -1, -1, 0], [-1, 4, 0, -1], [-1, 0, 4, -1], [0, -1, -1, 4]])
b = Matrix([[0], [0] ,[1] ,[1]])
print(f'精确解为{sspdm(copy.deepcopy(A), copy.deepcopy(b)).__str__(5, 5)}')
x = Matrix([[0], [0], [0], [0]])
print(f'3次迭代解为{sspdm(A, b, x, 3).__str__(5, 5)}')
print(A)
print(b)

t1 = Timer('sspdm(copy.deepcopy(A), copy.deepcopy(b))','from __main__ import sspdm, A, b, copy')
t2 = Timer('sspdm(A, b, x, 3)','from __main__ import sspdm, A, b, x, copy')
print(f'1000次精确求解所用时间为：{t1.timeit(number=1000)}')
print(f'1000次迭代求解所用时间为：{t2.timeit(number=1000)}')













