# 该文件定义了各种求解矩阵本征值问题的方法
# 运行1、2题作业代码时需将各题代码文件和该文件放在同一目录下

import itertools
from math import sqrt
from copy import deepcopy
from matrix import Matrix, I_Matrix, Jacobi_Matrix

def sign(x):        # 自定义符号函数
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0

def HHR(A:Matrix):      # Householder Hessenberg Reduction
    assert A.row == A.col, "行数不等于列数，不可约化"
    m = A.row
    U = I_Matrix(m)                     # 变换矩阵
    for k in range(1, m - 1):
        x = Matrix([[A[i, k]] for i in range(k+1, m+1)])
        e1 = Matrix([[1 if i == 0 else 0] for i in range(0, m-k)])  # 第一个元素为1的单位向量
        v = x + e1 * sign(x[1]) * sqrt(sum([t ** 2 for t in x]))    # 构造反射轴向量
        v = v * (1 / sqrt(sum([t ** 2 for t in v])))                # 归一化
        v *= v.reverse()                                            # 这里的v为一矩阵
        A1 = A.getSubMatrix((k+1, m), (k, m))       # 取出A_{k+1:m, k:m}
        B1 = v * A1                                 # 避免赋值时重复计算矩阵乘法
        for i, j in itertools.product(range(k + 1, m + 1), range(k, m + 1)):
            A[i, j] -= 2 * B1[i-k, j-k+1]           # k+1~m行的1～k-1列已经全部为0，不再需更新这些值
        A2 = A.getSubMatrix((1, m), (k+1, m))       # 取出A_{1:m, k+1:m}
        B2 = A2 * v
        for i, j in itertools.product(range(1, m + 1), range(k + 1, m + 1)):
            A[i, j] -= 2 * B2[i, j-k]               # 注意矩阵乘法的顺序！
        u = I_Matrix(m)                             # 记录每一步变换
        for i, j in itertools.product(range(k + 1, m + 1), range(k + 1, m + 1)):
            u[i, j] -= 2 * v[i - k, j - k]
        U *= u
    return A, U             # U^{T} A U = H

def GHR(A:Matrix):      # Givens Hessenberg Reduction
    n = A.row
    G = I_Matrix(n)                 # 变换矩阵
    for j in range(1, n-1):
        for i in range(n, j+1, -1):
            if abs(A[i, j]) > 10 ** -8:
                c, s = A[i-1, j] / sqrt(A[i-1, j]**2 + A[i, j]**2), A[i, j] / sqrt(A[i-1, j]**2 + A[i, j]**2)
                g = Jacobi_Matrix(n, i-1, i, c, s)          # 记录每一步变换
                G = G * g.reverse()
                A = g * A * g.reverse()
    return A, G             # G^{T} A G = H

def GHR_QR(A:Matrix):      # Givens Transformation——QR
    n = A.row
    G = I_Matrix(n)                 # 变换矩阵
    for j in range(1, n):
        for i in range(n, j, -1):
            if abs(A[i, j]) > 10 ** -8:
                c, s = A[i-1, j] / sqrt(A[i-1, j]**2 + A[i, j]**2), A[i, j] / sqrt(A[i-1, j]**2 + A[i, j]**2)
                g = Jacobi_Matrix(n, i - 1, i, c, s)  # 记录每一步变换
                G = G * g.reverse()
                A = g * A
    return G, A             # Q、R

def QRD(A:Matrix, tol=10**-5, im=500):      # QR-Decomposition 一般性QR分解
    over = lambda A: max(abs(A[i, i+1]) for i in range(1, A.row))
    A, G = HHR(A)       # 先进行Householder变换约化为上Hessenberg矩阵
    i = 0
    while over(A) > tol and i < im:         # 达到精度要求或迭代次数上限即终止
        Q, R = GHR_QR(A)
        A = R * Q
        G *= Q
        i += 1
    return A, G         # G^{T} A G = R, 其中R为拟上三角矩阵

def Wilkinson(T:Matrix):      # T为对称三对角矩阵
    n = T.row
    d = (T[n - 1, n - 1] - T[n, n]) / 2
    if d:   # 最后两个对角元不等，可取Wilkinson位移
        u = T[n, n] - T[n, n - 1] ** 2 / (d + sign(d) * sqrt(d ** 2 + T[n, n - 1] ** 2))  # 计算Wilkinson位移
        if abs(u - T[1, 1]) < 10 ** -20:            # 为避免算法中断，判断u和第一个对角元的接近程度
            u = T[n, n]  # 改取a_n
    else:   # 最后两个对角元相等，不能将d放在分母上
        u = T[n, n] if T[n, n] - T[1, 1] else 0     # 为了算法稳定性，放弃位移
    while True:
        success = 1              # 算法是否遭遇中断
        Tc = deepcopy(T)
        Z = I_Matrix(n)
        for k in range(1, n):
            x, z = Tc[k, k] - u, Tc[k+1, k]         # 不能为0，否则算法将中断
            if max(abs(x), abs(z)) == 0:
                success = 0
                break
            c, s = x / sqrt(x**2 + z**2) , -z / sqrt(x**2 + z**2)       # 计算cs对
            G = Jacobi_Matrix(n, k, k+1, c, s)                          # Givens旋转矩阵
            Tc = G.reverse() * Tc * G               # 对T做正交相似变换
            Z *= G                                  # 更新变换矩阵Z
        if success:             # 算法没有中断
            return Tc, Z        # Z^{T} T Z 仍为对称三对角矩阵
        else:
            u += 0.01           # 改变位移，重复以上计算

def QR_Wilkinson(A:Matrix, method=1, tol=10**-10):     # 隐式QR算法，对称三对角矩阵QR分解优先使用Wilkinson位移
    n = A.row           # n=A的行数=A的列数
    if method:
        T, Q = HHR(A)       # T为对称三对角矩阵，Q为正交变换矩阵，上海森堡约化优先使用Householder变换
    else:
        T, Q = GHR(A)       # T为对称三对角矩阵，Q为正交变换矩阵，上海森堡约化使用Givens变换
    q = 0
    while True:             # q=n-1即表示完全对角化
        for i in range(n-q-1, 0, -1):
            if abs(T[i, i+1]) < tol:        # Tol可变
                T[i, i+1], T[i+1, i] = 0, 0     # 设为0
                q += 1
            else:           # 已找出最大的q
                break
        if q == n - 1:      # 已完全对角化
            break
        p = 0               # 初始值置为0
        for i in range(n-q-2, 0, -1):
            if abs(T[i, i+1]) < tol:        # Tol可变
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
    return [T[i, i] for i in range(1, n+1)], Q     # T为对角矩阵，Q为正交变换矩阵

def power_method_max(A:Matrix, v:Matrix, tol = 10**-3, im = 200):     # 幂法找出按模最大的本征值和本征矢
    u = v
    lam = 0
    i = 0
    while (v - A*u).norm(0) > tol and i < im:
        v = A * u
        lam = v.norm(0)         # 主特征值近似值
        u = v * (1/lam)         # 规格化
        i += 1
    return lam, u               # 返回最大特征值和特征向量

def lud(A:Matrix):      # 矩阵的LU分解 (LU-decomposition)
    for k in range(1, A.row):
        if A[k, k] == 0:
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

def power_method_min(A:Matrix, v:Matrix, tol=10**-3, im=200):     # 反幂法找出按模最小的本征值和本征矢
    u = v
    lam = 0
    B = lud(deepcopy(A))          # 对A进行LU分解，为不改变原矩阵，将结果存入B中
    i = 0
    while (v - solve_LU(B, u)).norm(0) > tol and i < im:
        v = solve_LU(B, u)        # v = A^{-1}u
        lam = 1 / v.norm(0)       # 最小特征值近似值
        u = v * lam               # 规格化
        i += 1
    return lam, u

def Jacobi(A:Matrix, tol=10**-8):       # 经典雅可比算法
    n = A.row
    V = I_Matrix(n)           # 单位矩阵
    while True:
        p, q = 1, 2
        for i in range(1, n + 1):
            for j in range(i + 1, n + 1):
                if abs(A[i, j]) > abs(A[p, q]):
                    p, q = i, j
        if abs(A[p, q]) < tol:     # 已达到精度要求，跳出循环
            break
        r = (A[q, q] - A[p, p]) / (2 * A[p, q])
        t = 1 / (r + sqrt(1 + r**2)) if r >= 0 else -1 / (-r + sqrt(1 + r**2))
        c = 1 / sqrt(1 + t**2)
        s = t * c                   # 计算(c, s)对
        J = Jacobi_Matrix(n, p, q, c, s)
        A = J.reverse() * A * J
        V = V * J
    return [A[i, i] for i in range(1, n+1)], V      # 本征值和变换矩阵

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
    while u - l > 10**-5:
        if csn(a, b, r) >= m:
            u = r
        else:
            l = r
        r = (l + u) / 2
    return r
