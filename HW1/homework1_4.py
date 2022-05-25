# 计算物理第一次大作业第4题源代码
# 运行该程序大约需要200s，其中数值求解高斯点和求积系数需要较长时间
# 运行过程中图像会自动关闭

pause_time = 5      # pause_time为每张图片的显示时间，默认为5s，可调

# 以下导入的除matplotlib外都是python内置库
import copy
import time
from math import sin, sqrt, e, pi
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Songti SC']
plt.rcParams['axes.unicode_minus']=False


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

def diff(f, x, n=1):        # 默认求一阶导数
    h = 10 ** (n-6)
    if n == 1:
        return (f(x+h) - f(x-h)) / (2*h)
    else:
        return (diff(f, x+h, n-1) - diff(f, x-h, n-1)) / (2*h)

def Legendre(l, x):             # 勒让德多项式，l=0,1,2,···
    if l == 0:
        return 1
    elif l == 1:
        return x
    else:
        return ((2*l-1) * x * Legendre(l-1, x) - (l-1) * Legendre(l-2, x)) / l

def Legendre_diff(l, x):        # 勒让德多项式的导数，l=0,1,2,···
    if l == 0:
        return 0
    else:
        return (l * Legendre(l-1, x) - l * x * Legendre(l, x)) / (1 - x**2)

def basic_func(lx:list, k, x):         # 拉格朗日插值基函数
    l = 1
    for t in lx:
        l *= 1 if t == lx[k] else (x - t) / (lx[k] - t)
    return l

def complex_cotes(f, a, b, n):          # 复化柯特斯积分公式
    h = (b - a) / n
    x1 = [f(a + (i+1/4)*h) for i in range(n)]
    x2 = [f(a + (i+1/2)*h) for i in range(n)]
    x3 = [f(a + (i+3/4)*h) for i in range(n)]
    x4 = [f(a + i*h) for i in range(1, n)]
    C = h / 90 * (7*f(a) + 32*sum(x1) + 12*sum(x2) + 32*sum(x3) + 14*sum(x4) + 7*f(b))
    return C

def Newton_Method(f, x0):       # 牛顿法找x0附近的零点
    diff = lambda f,x: (f(x+(10**-5)) - f(x-(10**-5))) / (2*(10**-5))   # 差商型求导
    x = x0
    while abs(f(x0)) > (10 ** -6) or abs(x0 - x) > (10 ** -6):
        x = x0 - f(x0) / diff(f, x0)     # x存储的是x_{k+1}
        x0, x = x, x0       # 交换后，x0代表迭代得到的最新结果，x代表上一部迭代结果
    return x0

def Gauss_coefficient(n):       #   生成高斯点和求积系数
    f = lambda x: Legendre(n, x)        # 给定n，生成n阶勒让德多项式
    lp = [Newton_Method(f, -1)]         # lp存储n个高斯点
    for _ in range(n-1):
        def g(x):
            if x in lp:
                g_value = Legendre_diff(n, x)       # 洛必达法则
                for t in lp:
                    g_value /= (x - t) if t != x else 1     # 除掉已得到的因子
            else:
                g_value = Legendre(n, x)
                for t in lp:
                    g_value /= (x - t)      # 除掉已得到的因子
            return g_value
        lp.append(Newton_Method(g, -1))
    lc = []                             # lc存储n个求积系数
    for i in range(n):
        h = lambda x: basic_func(lp, i, x)
        C = complex_cotes(h, -1, 1, 64)      # 使用64阶cotes公式计算积分值
        lc.append(C)
    return lp, lc

def solve(k0, gamma, a, lp, lc):      # 求解矩阵方程
    mid = 10
    E0 = k0 ** 2
    n = len(lp)           # 高斯点的个数
    N = 2*n               # 对N个点做高斯求积
    lw = [E0 / 2 * lc[i] for i in range(n)] + [2 * (E0 + mid) / (1-lp[i])**2 * lc[i] for i in range(n)]    # 存储w_j，其中j=1,2,...,N
    lE = [E0 / 2 * (1 + lp[i]) for i in range(n)] + [(2*E0 + mid*(1+lp[i])) / (1-lp[i]) for i in range(n)]
    lE.append(E0)       # 存储E_j，其中j=1,2,...,N
    lD = [lw[i] / (E0 - lE[i]) for i in range(N)]                   # 存储D_j，其中j=1,2,...,N
    lD.append(complex(0, -pi))
    func_delta = lambda i, j: 1 if i == j else 0
    func_V = lambda i, j: 4*gamma * sin(sqrt(lE[i-1])*a) * sin(sqrt(lE[j-1])*a) / (lE[i-1]*lE[j-1])**(1/4)
    F = Matrix((N+1, N+1))                                          # 系数矩阵F
    V = Matrix([[func_V(i, N+1)] for i in range(1, N+2)])           # 右端向量[V]
    for i in range(1, N+2):
        for j in range(1, N+2):
            F[i, j] = func_delta(i, j) - lD[j - 1] * func_V(i, j)
    T = cpe(F, V)           # 列主元消去法解方程组
    return T                # 返回待求向量[T]


time_start = time.time()      # 开始计时


# -----------------------第一问-----------------------
print('------------矩阵方程的求解------------')
n = 25              # 决定高斯求积的精度
lp, lc = Gauss_coefficient(n)       # 生成高斯点和原始求积系数

gamma = 5
a = 1
k0 = 1
T = solve(k0, gamma, a, lp, lc)
print(f'k0 = {k0}, a = {a}, gamma = {gamma}')
print(f'使用{n}阶Gauss-Legendre多项式生成高斯点，求解矩阵方程的结果为：[T] =')
print(T)


# -----------------------第二问-----------------------
print('------------微分散射截面随k_0的变化------------')
print('结果如图所示')
lk0, lT = [i/50 for i in range(1, 501)], []
for k0 in lk0:
    T0 = solve(k0, gamma, a, lp, lc)[2*n+1]
    lT.append((abs(T0)) ** 2)
plt.plot(lk0, lT, label = r'$|\langle E_0|T|E_0 \rangle|^2 - k_0$')
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')      #设置上边和右边无边框
plt.xlabel(r'$k_0$')
plt.ylabel(r'$|\langle E_0|T|E_0 \rangle|^2$')
plt.legend(loc='upper left', frameon=True)
plt.title(rf'$|\langle E_0|T|E_0 \rangle|^2 - k_0$变化曲线 ($a = 1, \gamma = {gamma}$)')
plt.ion()
plt.pause(pause_time)
plt.close()


# -----------------------第三问-----------------------
print('------------sin^2(delta_0)~k变化关系------------')
print('如图所示')
gamma_h2 = 25
a = 1
lk, ld = [i/500 for i in range(1, 5001)], []
for k in lk:
    j = complex(0, 1)
    r = ((1 + gamma_h2 / k * e**(-k*a*j) * sin(k*a)) / (1 + gamma_h2 / k * e**(k*a*j) * sin(k*a))).real
    ld.append((1 - r) / 2)

plt.plot(lk, ld, label = r'$|\sin \delta_0|^2 - k$')
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
plt.xlabel(r'$k$')
plt.ylabel(r'$|\sin \delta_0|^2$')
plt.legend(loc='upper left', frameon=True)
plt.title(rf'$|\sin \delta_0|^2 - k$变化曲线 ($a = 1, \gamma/\hbar^2 = {gamma_h2}$)')
plt.ion()
plt.pause(pause_time)
plt.close()


# -----------------------第四问-----------------------
print('------------gamma很大时共振点位置------------')
print('如图所示，共振点位置满足：ka = n\pi')
gamma_h2 = 200
a = 1
lk, ld = [i/500 for i in range(1, 5001)], []
for k in lk:
    j = complex(0, 1)
    r = ((1 + gamma_h2 / k * e**(-k*a*j) * sin(k*a)) / (1 + gamma_h2 / k * e**(k*a*j) * sin(k*a))).real
    ld.append((1 - r) / 2)

plt.plot(lk, ld, label = r'$|\sin \delta_0|^2 - k$')
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
plt.xlabel(r'$k$')
plt.ylabel(r'$|\sin \delta_0|^2$')
plt.legend(loc='upper left', frameon=True)
plt.title(rf'$|\sin \delta_0|^2 - k$变化曲线 ($a = 1, \gamma/\hbar^2 = {gamma_h2}$)')
plt.ion()
plt.pause(pause_time)
plt.close()


time_end = time.time()      # 结束计时
print(f'运行结束，耗时{time_end - time_start}s')

