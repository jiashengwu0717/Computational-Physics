# 计算物理第三次大作业第3题源代码
# 运行该程序大约需要20min，谐振子势的演化过程计算较慢
# 对无限深势井和谐振子势，概率分布演化计算结束后，将定格显示最终的结果


import time
import math
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
plt.rcParams['font.sans-serif'] = ['Songti SC']
plt.rcParams['axes.unicode_minus'] = False


time_start = time.time()      # 开始计时


# ------------------------------ 无限深势井 -------------------------------
# 模拟参数1
V = lambda x: 0 if 0 <= x <= 15 else math.inf  # 无限深势井
sigma0 = 0.5                    # 波包展宽
k0 = 17 * math.pi               # 初始动量
D_x = 0.02                      # 空间步长
D_t = 1 / 2 * D_x ** 2          # 时间步长
alpha = D_t / (2 * D_x ** 2)

x = np.arange(0, 15.01, D_x)    # 网格点
psi = np.exp(-1/2 * ((x-5)/sigma0)**2 + 1j * k0 * x)    # 初态波函数
R1 = psi.real                   # 实部
I1 = psi.imag                   # 虚部
R1[0], R1[-1] = 0, 0            # 边界处取为0
I1[0], I1[-1] = 0, 0            # 边界处取为0
for i in range(1, len(x) - 1):  # 将虚步推进1/2个时间步长
    I1[i] += ((R1[i + 1] + R1[i - 1] - 2 * R1[i]) / (D_x ** 2) - V(x[i]) * R1[i]) * D_t / 2
R2 = np.zeros(len(x))
I2 = np.zeros(len(x))

t = 0                           # 初始时刻
plt.ion()                       # 打开交互模式
while t < 2:                    # 演化 2s
    plt.cla()                   # 清除上一次绘图
    for i in range(1, len(x) - 1):          # 更新波函数
        R2[i] = R1[i] - 2 * alpha * (I1[i + 1] + I1[i - 1]) + (4 * alpha + V(x[i]) * D_t) * I1[i]
    t += D_t / 2
    y = I1 ** 2 + R2 * R1
    plt.plot(x, y, c='blue')                # 绘制概率分布图
    plt.ylim(0, 1)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$\rho$', rotation=0)
    plt.title(f't = {round(t, 2)} s')
    plt.pause(D_t)
    for i in range(1, len(x) - 1):          # 更新波函数
        I2[i] = I1[i] + 2 * alpha * (R2[i + 1] + R2[i - 1]) - (4 * alpha + V(x[i]) * D_t) * R2[i]
    t += D_t / 2
    y = R2 ** 2 + I2 * I1
    plt.plot(x, y, c='blue')                # 绘制概率分布图
    plt.ylim(0, 1)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$\rho$', rotation=0)
    plt.title(f't = {round(t, 2)} s')
    plt.pause(D_t)
    R1, I1 = deepcopy(R2), deepcopy(I2)     # 推进到下一步
plt.ioff()                      # 关闭交互模式
plt.show()                      # 定格显示最后一张图


# -------------------------------- 谐振子势 ---------------------------------
# 模拟参数2
V = lambda x: 5 * x ** 2            # 一维谐振子势
sigma0 = 0.5                        # 初始波包展宽
k0 = 3 * math.pi                    # 初始动量
D_x = 0.02                          # 空间步长
D_t = 1 / 4 * D_x ** 2              # 时间步长
alpha = D_t / (2 * D_x ** 2)

x = np.arange(-12, 12.01, D_x)      # 网格点
Vx = V(x)                           # 相应格点上的势场
psi = np.exp(-1/2 * ((x-5)/sigma0)**2 + 1j * k0 * x)    # 初态波函数
R1 = psi.real                       # 实部
I1 = psi.imag                       # 虚部
for i in range(1, len(x) - 1):      # 将虚步推进1/2个时间步长
    I1[i] += ((R1[i + 1] + R1[i - 1] - 2 * R1[i]) / (D_x ** 2) - Vx[i] * R1[i]) * D_t / 2
R2 = np.zeros(len(x))
I2 = np.zeros(len(x))

t = 0                           # 初始时刻
plt.ion()                       # 打开交互模式
while t < 3:                    # 演化3s
    plt.cla()                   # 清除上一次绘图
    for i in range(1, len(x) - 1):       # 更新波函数
        R2[i] = R1[i] - 2 * alpha * (I1[i + 1] + I1[i - 1]) + (4 * alpha + Vx[i] * D_t) * I1[i]
    t += D_t / 2
    y = I1 ** 2 + R2 * R1
    plt.plot(x, y, c='red')              # 绘制概率分布图
    plt.ylim(0, 1.1)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$\rho$', rotation=0)
    plt.title(f't = {round(t, 2)} s')
    plt.pause(D_t)
    for i in range(1, len(x) - 1):       # 更新波函数
        I2[i] = I1[i] + 2 * alpha * (R2[i + 1] + R2[i - 1]) - (4 * alpha + Vx[i] * D_t) * R2[i]
    t += D_t / 2
    plt.plot(x, y, c='red')              # 绘制概率分布图
    plt.ylim(0, 1.1)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$\rho$', rotation=0)
    plt.title(f't = {round(t, 2)} s')
    plt.pause(D_t)
    R1, I1 = deepcopy(R2), deepcopy(I2)  # 推进到下一步
plt.ioff()                      # 关闭交互模式
plt.show()                      # 定格显示最后一张图


time_end = time.time()          # 结束计时
print(f'运行结束，耗时{time_end - time_start}s')
