from copy import deepcopy
from math import sqrt
import numpy as np
from matrix import Matrix, I_Matrix, Jacobi_Matrix
import itertools


def sign(x):        # 自定义符号函数
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0

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
    return A            # L0和U的信息存入A中

def solve_LU(A:Matrix, b:Matrix):      # A中存储了矩阵L和U，且解方程不改变A和b
    w = deepcopy(b)
    for i in range(1, A.row + 1):       # 单位下三角矩阵
        for j in range(1, i):
            w[i] -= A[i, j] * w[j]
    for i in range(A.row, 0, -1):
        for j in range(A.col, i, -1):
            w[i] -= A[i, j] * w[j]
        w[i] /= A[i, i]
    return w


def HHR(A:Matrix):      # Householder Hessenberg Reduction
    assert A.row == A.col, "行数不等于列数，不可约化"
    m = A.row
    U = I_Matrix(m)                     # 变换矩阵
    for k in range(1, m - 1):
        x = Matrix([[A[i, k]] for i in range(k+1, m+1)])
        e1 = Matrix([[1 if i == 0 else 0] for i in range(0, m-k)])
        v = x + e1 * sign(x[1]) * sqrt(sum([t ** 2 for t in x]))    # x[1]是0，如何处理？
        v = v * (1 / sqrt(sum([t ** 2 for t in v])))
        v *= v.reverse()                                            # 这里的v代表一矩阵
        A1 = Matrix([A[i][k-1:m] for i in range(k+1, m+1)])         # 取出A_{k+1:m, k:m}
        B1 = v * A1                             # 避免赋值时重复计算矩阵乘法
        for i, j in itertools.product(range(k + 1, m + 1), range(k, m + 1)):
            A[i, j] -= 2 * B1[i-k, j-k+1]       # k+1~m行的1～k-1列已经全部为0，不再需更新这些值
        A2 = Matrix([A[i][k:m] for i in range(1, m+1)])             # 取出A_{1:m, k+1:m}
        B2 = A2 * v
        for i, j in itertools.product(range(1, m + 1), range(k + 1, m + 1)):
            A[i, j] -= 2 * B2[i, j-k]           # 注意矩阵乘法的顺序！
        u = I_Matrix(m)                 # 记录每一步变换
        for i, j in itertools.product(range(k + 1, m + 1), range(k + 1, m + 1)):
            u[i, j] -= 2 * v[i - k, j - k]
        U *= u
    return A, U         # U^{T} A U = H

print('----------Householder变换将一般矩阵约化为上海森堡矩阵----------')
A = Matrix([[1, -2, 2, 4], [-2, 3, 4, 1], [2, 2, -1, -5], [3, -6, 2, -2]])
B = Matrix([[5,-3,2], [6,-4,4], [4,-4,5]])
print(B)
print('------->')
r = HHR(deepcopy(B))
print(r[0].__str__(4))
print((r[1].reverse() * B * r[1]).__str__(4))

print('')

def GHR(A:Matrix):      # Givens Hessenberg Reduction——QR
    n = A.row
    G = I_Matrix(n)                 # 变换矩阵
    for j in range(1, n):
        for i in range(n, j, -1):
            if A[i, j]:
                c, s = A[i-1, j] / sqrt(A[i-1, j]**2 + A[i, j]**2), A[i, j] / sqrt(A[i-1, j]**2 + A[i, j]**2)
                g = Jacobi_Matrix(n, i - 1, i, c, s)  # 记录每一步变换
                G = G * g.reverse()
                A = g * A
    return G, A             # Q、R

print('----------Givens变换将一般矩阵约化为上海森堡矩阵----------')
C = Matrix([[4, 1, 0, -1], [1, 4, -1, 0], [0, -1, 4, 1], [-1, 0, 1, 4]])
B = Matrix([[1,3,-1], [2,-5,1], [4,1,0]])
print(B)
Q, R = GHR(B)
print('--------->')
print(f'{Q.__str__(5)}')
print(f'* {R.__str__(5)}')
print(f'= {(Q * R).__str__(5)}')
print('')


def QRD(A:Matrix):         # QR-Decomposition 一般性QR分解
    over = lambda A: max([abs(A[i, j]) for i in range(1, A.row) for j in range(i+1, A.col+1)])
    # while over(A) > 10**-5:
    for _ in range(100):
        Q, R = GHR(A)
        A = R * Q
    return A

def QRsymmetry(A:Matrix):       # 实对称矩阵的实用QR算法（带位移）
    A = HHR(A)[0]
    k = A.row
    while k > 1:
        if abs(A[k, k-1]) == 0:         # 精度要求不高时，可以将绝对值小于e作为判停标准
            k -= 1
        s = A[k, k]
        for j in range(1, k + 1):
            A[j, j] -= s
        Ak = Matrix([A[i][:k] for i in range(1, k+1)])      # 取出A_{1:k, 1:k}
        Q, R = GHR(Ak)
        Ak = R * Q
        for i, j in itertools.product(range(1, k + 1), range(1, k + 1)):
            A[i, j] = Ak[i, j]
        for j in range(1, k + 1):
            A[j, j] += s
    return A

print('------------------QR分解-------------------')
A = Matrix([[3, 17, -37, 18, -40],
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0 ,0],
            [0, 0, 0, 1 ,0]])
print(QRD(A).__str__(5))
B = np.array([[3, 17, -37, 18, -40],
              [1, 0, 0, 0, 0],
              [0, 1, 0, 0, 0],
              [0, 0, 1, 0 ,0],
              [0, 0, 0, 1 ,0]])
print(B)
print(np.linalg.eig(B)[0])
print('')
A = Matrix([[1,1,1,1], [1,2,3,4], [1,3,6,10], [1,4,10,20]])
# print(QRsymmetry(A).__str__(10, 15))
print('')


def power_method_max(A:Matrix, v:Matrix):     # 幂法找出按模最大的本征值和本征矢
    u = v
    lam = 0
    while (v - A*u).norm(0) > 10 ** -10:
        v = A * u
        lam = v.norm(0)         # 主特征值近似值
        u = v * (1/lam)         # 规格化
    return lam, u               # 返回最大特征值和特征向量

def power_method_min(A:Matrix, v:Matrix):     # 反幂法找出按模最小的本征值和本征矢
    u = v
    lam = 0
    B = lud(deepcopy(A))          # 对A进行LU分解，为不改变原矩阵，将结果存入B中
    while (v - solve_LU(B, u)).norm(0) > 10 ** -5:
        v = solve_LU(B, u)        # v = A^{-1}u
        lam = 1 / v.norm(0)       # 最小特征值近似值
        u = v * lam               # 规格化
    return lam, u

print('----------------------幂法与反幂法-----------------------')
A = Matrix([[3, 1], [1, 3]])
v = Matrix([[0], [1]])
print(power_method_max(A, deepcopy(v))[0])
print(power_method_min(A, deepcopy(v))[0])
print(power_method_min(A, deepcopy(v))[1])
print('')


def Arnodi(A:Matrix):   # Arnodi算法
    m = A.row
    H, Q = Matrix((m, m)), Matrix((m, m))
    Q[1, 1] = 1
    for j in range(1, m+1):
        r = A * Q.getCol(j)
        for i in range(1, j+1):
            H[i, j] = (Q.getCol(i).reverse() * r)[1, 1]   # 数据类型转化为整数或浮点数
            r -= Q.getCol(i) * H[i, j]
        if j == m:              # 结束
            return H, Q         # Q^{T} A Q = H
        H[j+1, j] = r.norm(2)
        if H[j+1, j] == 0:
            continue
        for i in range(1, m+1):
            Q[i, j+1] = r[i] / H[j+1, j]

def Lanzcos(A:Matrix):  # Lanzcos算法
    m = A.row
    T, Q = Matrix((m, m)), Matrix((m, m))
    Q[1, 1] = 1
    for k in range(1, m+1):
        u = A * Q.getCol(k)
        T[k, k] = (Q.getCol(k).reverse() * u)[1, 1]
        if k == m:              # 结束
            return T, Q         # # Q^{T} A Q = T
        u -= Q.getCol(k) * T[k, k] if k == 1 else Q.getCol(k-1) * T[k, k-1] + Q.getCol(k) * T[k, k]
        T[k, k+1], T[k+1, k] = u.norm(2), u.norm(2)
        if T[k, k+1] == 0:
            break
        for i in range(1, m+1):
            Q[i, k+1] = u[i] / T[k, k+1]


print('----------------krylov子空间迭代法----------------')
A = Matrix([[1, -2, 2, 4], [-2, 3, 4, 1], [2, 2, -1, -5], [3, -6, 2, -2]])
print(A)
print('------->')
H = Arnodi(A)[0]
print(H)
print('')
B = Matrix([[5,-3,2], [6,-4,4], [4,-4,5]])
print(B)
print('------->')
r = Arnodi(B)
print(r[0].__str__(3))
print((r[1].reverse() * B * r[1]).__str__(3))
print('')
A = Matrix([[1,1,1,1], [1,2,3,4], [1,3,6,10], [1,4,10,20]])
print(A)
print('------->')
r = Lanzcos(A)
print(r[0].__str__(3))
print((r[1].reverse() * A * r[1]).__str__(3))
print('')



def Jacobi(A:Matrix):       # 经典雅可比算法
    n = A.row
    V = I_Matrix(n)           # 单位矩阵
    while True:
        p, q = 1, 2
        for i in range(1, n + 1):
            for j in range(i + 1, n + 1):
                if abs(A[i, j]) > abs(A[p, q]):
                    p, q = i, j
        if abs(A[p, q]) < 10 ** -10:     # 已达到精度要求，跳出循环
            break
        r = (A[q, q] - A[p, p]) / (2 * A[p, q])
        t = 1 / (r + sqrt(1 + r**2)) if r >= 0 else -1 / (-r + sqrt(1 + r**2))
        c = 1 / sqrt(1 + t**2)
        s = t * c                   # 计算(c, s)对
        J = Jacobi_Matrix(n, p, q, c, s)
        A = J.reverse() * A * J
        V = V * J
    return sorted([A[i, i] for i in range(1, n+1)], reverse=True), V

print('---------------经典雅可比算法----------------')
A = Matrix([[1,1,1,1], [1,2,3,4], [1,3,6,10], [1,4,10,20]])
HHR(A)
print(A.__str__(4))
print(Jacobi(A)[0])

B = np.array([[1,1,1,1], [1,2,3,4], [1,3,6,10], [1,4,10,20]])
print(B)
print(np.linalg.eig(B)[0])
print('')


def csn(a:list, b:list, x):    # change sign number
    count = 0
    q = a[0] - x
    for k in range(1, len(a) + 1):
        if q < 0:
            count += 1          # 变号，计时器+1
        if k < len(a):
            if q == 0:          # 使递推可以进行下去
                q = 10**-15
            q = a[k] - x - (b[k])**2 / q
    return count        # 返回变号数

def dichotomy(A:Matrix, m):    # 对称三对角矩阵的二分法
    a = [A[i, i] for i in range(1, A.row + 1)]
    b = [0] + [A[i+1, i] for i in range(1, A.row)]
    l = -A.norm(0)
    u = -l
    r = 0
    while u - l > 10**-10:
        if csn(a, b, r) >= m:
            u = r
        else:
            l = r
        r = (l + u) / 2
    return r

print('--------------二分法求对称三对角矩阵从小到大第m个特征值---------------')
A = Matrix([[1,1,1,1], [1,2,3,4], [1,3,6,10], [1,4,10,20]])
HHR(A)
print(A.__str__(5))
for i in range(1, 5):
    print(dichotomy(A, i))
print('')


print('------------------------分而治之-------------------------')
def divide_conquer(A):      # 分而治之法
    pass


def Wilkinson(T:Matrix):        # T为三对角矩阵
    n = T.row
    Z = I_Matrix(n)
    d = (T[n-1, n-1] - T[n, n]) / 2
    u = T[n, n] - T[n, n-1]**2 / (d + sign(d)*sqrt(d**2 + T[n, n-1]**2))    # 计算Wilkinson位移
    for k in range(1, n):
        x, z = T[k, k] - u , T[k+1, k]
        c, s = x / sqrt(x**2 + z**2) , z / sqrt(x**2 + z**2)        # 计算cs对
        G = Jacobi_Matrix(n, k, k+1, c, -s)                         # Givens旋转矩阵
        T = G.reverse() * T * G                     # 对T做正交相似变换
        Z *= G                                      # 更新变换矩阵Z
    return T, Z                 # Z^{T} T Z 仍为对称三对角矩阵


def QR_Wilkinson(A:Matrix):     # 隐式QR算法，QR分解使用Wilkinson位移
    n = A.row           # n=A的行数=A的列数
    T, Q = HHR(A)       # T为对称三对角矩阵，Q为正交变换矩阵
    q = 0
    while True:             # q=n-1即表示完全对角化
        for i in range(n-q-1, 0, -1):
            if abs(T[i, i+1]) < 10**-10:        # Tol可变
                T[i, i+1], T[i+1, i] = 0, 0     # 设为0
                q += 1
            else:           # 已找出最大的q
                break
        if q == n - 1:      # 已完全对角化
            break
        p = 0               # 初始值置为0
        for i in range(n-q-2, 0, -1):
            if abs(T[i, i+1]) < 10**-10:        # Tol可变
                T[i, i+1], T[i+1, i] = 0, 0     # 设为0
                p = i       # 最小的p
                break
        D22 = T.getSubMatrix((p+1, n-q), (p+1, n-q))    # 取出不可约子矩阵
        Z = Wilkinson(D22)[1]
        G = I_Matrix(n)
        for i, j in itertools.product(range(1, n-p-q+1), range(1, n-p-q+1)):
            G[p + i, p + j] = Z[i, j]
        T = G.reverse() * T * G
        Q *= G
    return T, Q



print('-----------------实对称矩阵的隐式QR算法-------------------')
A = Matrix([[1,2,0,0], [2,3,4,0], [0,4,5,6], [0,0,6,7]])
print(A)
re = QR_Wilkinson(deepcopy(A))
print(re[0])
print(re[1])
print(re[1].reverse() * A * re[1])

print('-----------------------------------')

A = Matrix([[1.95,0,0,0],[0,0.95,1,0],[0,1,0.95,0],[0,0,0,1.95]])
print(A)
B = A.inverse()
v = Matrix([[1], [-2], [0], [-1]])


