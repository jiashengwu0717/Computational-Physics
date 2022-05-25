# 计算物理第三次大作业第1题源代码
# 运行该程序大约需要20min，且最终结果受初始条件的影响


import time
from matrix import Matrix
from math import sqrt, log
from functools import reduce
from random import uniform
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Songti SC']
plt.rcParams['axes.unicode_minus'] = False


class generate_A(Matrix):       # 生成系数矩阵A
    def __init__(self, N):
        matrix = [[-2 if r == c else 0 for c in range(N)] for r in range(N)]
        matrix[0][N - 1], matrix[N - 1][0] = 1, 1
        for i in range(N - 1):
            matrix[i + 1][i], matrix[i][i + 1] = 1, 1
        super(generate_A, self).__init__(matrix)


class generate_B(Matrix):       # 生成系数矩阵B
    def __init__(self, N):
        matrix = [[0 if r == c else 0 for c in range(N)] for r in range(N)]
        matrix[0][N - 1], matrix[N - 1][0] = 1, -1
        for i in range(N - 1):
            matrix[i + 1][i], matrix[i][i + 1] = 1, -1
        super(generate_B, self).__init__(matrix)


def Marsaglia():        # 生成满足高斯分布的随机数，一次同时生成2个
    while True:
        u, v = uniform(0, 1), uniform(0, 1)
        w = (2*u-1)**2 + (2*v-1)**2
        if w <= 1:
            break
    z = sqrt(-2 * log(w) / w)
    return np.array([(2*u-1) * z, (2*v-1) * z])


def solve_x(A:Matrix, b: np.array):     # 解方程
    n = len(b)
    x, y = np.zeros(n), np.zeros(n)
    y[0] = b[0]
    for i in range(1, n):
        y[i] = b[i] - A[i, i + 1] * y[i - 1]
    x[n - 1] = y[n - 1] / A[n, n]
    for i in range(n - 2, -1, -1):
        x[i] = y[i] / A[i + 1, i + 1] - A[i + 1, i + 2] * x[i + 1]
    return x


time_start = time.time()      # 开始计时

# ---------------------------------- 输入参数 --------------------------------------

N = 40000           # 宏粒子数
n_0 = 10 ** 20      # 真实粒子密度
T = 10 ** 8         # 温度
c0 = 3 * 10**8                  # 真空中光速
epsilon = 8.85419 * 10**-12     # 真空中介电常数
kb = 1.38065 * 10**-23          # 玻尔兹曼常数
e = 1.60218 * 10**-19           # 基本电荷量
m_e = 9.10938 * 10**-31         # 电子质量
lambda_D = sqrt(epsilon * kb * T / (n_0 * e**2))
L_x = 150 * lambda_D    # 模拟盒长度
N_x = 400               # 网格数
D_x = L_x / N_x         # 每个网格的大小
D_t = D_x / c0          # 时间步长
Ts = 4 * L_x / c0       # 总的模拟时间

t = 0                   # 初始时刻
r1 = np.random.rand(N // 2) * L_x         # 创建长度为N/2，取值范围为[0,L_x)的均匀分布的随机数组
r2 = np.random.rand(N // 2) * L_x         # 创建长度为N/2，取值范围为[0,L_x)的均匀分布的随机数组
p1 = reduce(lambda x, y: np.append(x, y), [(Marsaglia() + 3 * sqrt(2)) * sqrt(m_e * kb * T) for _ in range(N//4)])
p2 = reduce(lambda x, y: np.append(x, y), [(Marsaglia() - 3 * sqrt(2)) * sqrt(m_e * kb * T) for _ in range(N//4)])
# 分别创建长度为N/2，取值满足麦克斯韦分布的随机数组

plt.scatter(r1 / lambda_D, p1 / (m_e * c0), c = 'red', s = 1)               # 电子在相空间的初始分布
plt.scatter(r2 / lambda_D, p2 / (m_e * c0), c = 'deepskyblue', s = 1)
plt.xlabel(r'$x / \lambda_D$')
plt.ylabel(r'$p_x / m_e c$')
plt.title('电子在相空间的初始分布')
plt.pause(0.1)

# ------------------------------------- 启动 ------------------------------------------

A = generate_A(N_x).getSubMatrix((1, N_x - 1), (1, N_x - 1))        # 系数矩阵A的n-1阶主子矩阵
for i in range(1, A.row + 1):                                       # 平方根法 A = L D L^{T} = U^{T} D U
    for j in range(1, i):
        for k in range(1, j):                       # Note: i > j > k
            A[i, j] -= A[i, k] * A[k, j]            # 计算t(i,j)，并存入A(i,j)中
        A[j, i] = A[i, j] / A[j, j]                 # 计算l(i,j)，并存入A(j,i)中
    for k in range(1, i):
        A[i, i] -= A[i, k] * A[k, i]                # 计算d(i)，并存入A(i,i)中

Nj = np.zeros(N_x)                              # Nj记录j格点的粒子数 ----------- 必须次次归零！
for x in np.append(r1, r2):
    index = int(x / D_x)                        # 粒子所属的较小网格编号
    Nj[index] += index + 1 - x / D_x            # 分配给较小的网格
    Nj[(index+1) % N_x] += x / D_x - index      # 分配给较大的网格，周期性边界条件
nj = Nj * N_x / N                               # j格点处的真实密度，暂时舍弃n_0
n0 = np.ones(N_x)                               # 正电荷密度，暂时舍弃 e/epsilon_0*n_0*(Dx)^2 ------- 不变量，只需第一次生成
phi = solve_x(A, (nj - n0)[:-1])                # 解出电势分布
phi = np.append(phi, 0.0).reshape(N_x, 1)       # 调整维数
B = np.matrix(generate_B(N_x).matrix)           # 生成由电势求电场的系数矩阵
Ej = - B * phi * (D_x / 2) * n_0 * e / epsilon  # Ej记录格点处电场分布
Ei1 = np.zeros(N // 2)                          # Ei1记录粒子束1感受到的电场 -------- 覆盖效果，只需第一次生成
for i in range(len(r1)):
    index = int(r1[i] / D_x)                    # 较小的网格编号
    Ei1[i] = (index + 1 - r1[i] / D_x) * Ej[index] + (r1[i] / D_x - index) * Ej[(index+1) % N_x]
p1 += Ei1 * e * D_t / 2                                 # 推进动量
v1 = p1 / (np.sqrt(1 + (p1 / (m_e * c0))**2) * m_e)     # 更新速度
r1 += (v1 * D_t)                                        # 推进位置
r1 %= L_x                                               # 周期性边条件
Ei2 = np.zeros(N // 2)                          # Ei2记录粒子束2感受到的电场 -------- 覆盖效果，只需第一次生成
for i in range(len(r2)):
    index = int(r2[i] / D_x)                    # 较小的网格编号
    Ei2[i] = (index + 1 - r2[i] / D_x) * Ej[index] + (r2[i] / D_x - index) * Ej[(index+1) % N_x]
p2 += Ei2 * e * D_t / 2                                  # 推进动量
v2 = p2 / (np.sqrt(1 + (p2 / (m_e * c0))**2) * m_e)      # 更新速度
r2 += (v2 * D_t)                                         # 推进位置
r2 %= L_x                                                # 周期性边条件
t += D_t                                        # 更新时间

# ------------------------------------- 循环模拟 --------------------------------------------

plt.ion()
while t < Ts:
    plt.cla()
    Nj = np.zeros(N_x)                  # 先归零 重要！！！
    for x in np.append(r1, r2):                     # 更新属于各格点的粒子数Nj
        index = int(x / D_x)                        # 较小的网格编号
        Nj[index] += index + 1 - x / D_x            # 分配给较小的网格
        Nj[(index + 1) % N_x] += x / D_x - index    # 分配给较大的网格，周期性边界条件
    nj = Nj * N_x / N                                       # j格点处的真实密度（暂时舍去n_0）
    phi = solve_x(A, (nj - n0)[:-1])                        # 解出电势分布
    phi = np.append(phi, 0.0).reshape(N_x, 1)               # 调整维数
    Ej = - B * phi * (D_x / 2) * n_0 * e / epsilon          # 各格点处的电场
    for i in range(len(r1)):             # 将网格点上的电场值分配到连续空间的粒子束1位置上
        index = int(r1[i] / D_x)         # 较小的网格编号
        Ei1[i] = (index + 1 - r1[i] / D_x) * Ej[index] + (r1[i] / D_x - index) * Ej[(index + 1) % N_x]
    p1 += Ei1 * e * D_t                                       # 推进动量
    v1 = p1 / (np.sqrt(1 + (p1 / (m_e * c0)) ** 2) * m_e)     # 更新速度
    r1 += v1 * D_t                                            # 推进位置
    r1 %= L_x                                                 # 周期性边条件
    for i in range(len(r2)):             # 将网格点上的电场值分配到连续空间的粒子束2位置上
        index = int(r2[i] / D_x)         # 较小的网格编号
        Ei2[i] = (index + 1 - r2[i] / D_x) * Ej[index] + (r2[i] / D_x - index) * Ej[(index + 1) % N_x]
    p2 += Ei2 * e * D_t                                       # 推进动量
    v2 = p2 / (np.sqrt(1 + (p2 / (m_e * c0)) ** 2) * m_e)     # 更新速度
    r2 += v2 * D_t                                            # 推进位置
    r2 %= L_x                                                 # 周期性边条件
    t += D_t                                             # 更新时间
    if int(t / D_t) % 10 == 0:                                              # 每推进10步更新一次
        plt.scatter(r1 / lambda_D, p1 / (m_e * c0), c='red', s=1)           # 在相空间作图
        plt.scatter(r2 / lambda_D, p2 / (m_e * c0), c='deepskyblue', s=1)
        plt.xlabel(r'$x / \lambda_D$')
        plt.ylabel(r'$p_x / m_e c$')
        plt.title(f'演化至：t = {round(t * 10**12, 1)} ps，进度：{int(t / D_t)} / {int(Ts / D_t)}')
        plt.pause(0.1)                                  # 显示时间0.1秒
plt.ioff()
plt.show()              # 定格显示最终演化结果


time_end = time.time()          # 结束计时
print(f'运行结束，耗时{time_end - time_start}s')
