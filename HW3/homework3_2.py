# 计算物理第三次大作业第2题源代码
# 运行该程序时间取决于输入的参数，若取alpha=25度，T/T_s=200，a=1，大约需要200s
# 运行过程中出现的图像会自动关闭

pause_time = 5          # pause_time为每张图片的显示时间，默认为5s，可调

import time
import math
from math import sin, cos, sqrt, pi
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Songti SC']
plt.rcParams['axes.unicode_minus'] = False


alpha = eval(input('请输入月球轨道平面与地球赤道面的夹角，单位为角度制：'))
n = eval(input('请输入模拟的总时间，单位为恒星日：'))
a = eval(input('请输入卫星初始圆周运动周期，单位为恒星日：'))
print('正在计算中...')

# ------------------------------------ 输入参数 --------------------------------------
G = 6.6743 * 10**-11        # 万有引力常数
M_e = 5.9736 * 10**24       # 地球质量
GM_e = G * M_e
T_s = 86164                 # 一个恒星日
r_m = 384400000             # 月地距离
T_m = 27.32 * T_s           # 月球绕地球运动周期
M_m = 7.3744 * 10**22       # 月球质量
GM_m = G * M_m
alpha = alpha * math.pi / 180       # 调整月球轨道平面与地球赤道面的夹角为弧度制


def func_m(t):      # 月球运动方程
    x = r_m * cos(alpha) * cos(2 * pi * t / T_m)
    y = r_m * sin(2 * pi * t / T_m)
    z = r_m * sin(alpha) * cos(2 * pi * t / T_m)
    return np.array([x, y, z])


def func_a(r:np.array, t):         # 计算加速度
    return - GM_e / sum(r ** 2) ** (3/2) * r - GM_m / sum((r - func_m(t)) ** 2) ** (3/2) * (r - func_m(t))


def simulation(n, a = 1.0):         # 模拟n个恒星日的变化，a为卫星周期参数
    T = a * T_s                                             # 卫星周期
    r_0 = (G * M_e * T ** 2 / (4 * pi ** 2)) ** (1 / 3)     # 卫星初始轨道半径
    v_0 = 2 * pi * r_0 / T                                  # 卫星初始运动速度
    t = 0                           # 初始时刻
    tau = 0.001 * a * T_s           # 时间步长：0.001 a T_s，为保证精度，步长随a进行调整
    tau2 = tau ** 2
    r = np.array([r_0, 0, 0])       # 初始位置
    v = np.array([0, v_0, 0])       # 初始速度
    lt = [0]                                # 时间
    lr = [sqrt(sum(r ** 2))]                # 记录r～t变化
    phi_change = 0                          # 将phi角调整为单调增加
    lphi = [math.atan(r[1] / r[0])]         # 记录phi～t变化
    ltheta = [math.asin(r[2] / lr[-1])]     # 记录theta～t变化
    while t < n * T_s:
        l1 = func_a(r, t)
        l2 = func_a(r + tau / 2 * v, t + tau / 2)
        l3 = func_a(r + tau / 2 * v + tau2 / 4 * l1, t + tau / 2)
        l4 = func_a(r + tau * v + tau2 / 2 * l2, t + tau)
        r += tau * v + tau2 / 6 * (l1 + l2 + l3)
        v += tau / 6 * (l1 + 2 * l2 + 2 * l3 + l4)
        t += tau
        lt.append(t)
        lr.append(sqrt(sum(r ** 2)))
        phi = math.atan(r[1] / r[0])
        if phi + phi_change < lphi[-1]:                 # phi角越过(k+1/2)pi的位置
            phi_change += math.pi                       # 调整pi，使phi角单调增加
        lphi.append(phi + phi_change)
        ltheta.append(math.asin(r[2] / lr[-1]))
    lDr = [(r - r_0) / 1000 for r in lr]                # 径向偏移量，以km为单位
    lDphi = [lphi[k] - 2 * math.pi * lt[k] / T_s for k in range(len(lt))]      # 和地球自转角度比较
    lt = [t / T_s for t in lt]                          # 以T_s为单位
    return lt, lDr, lDphi, ltheta


time_start = time.time()      # 开始计时

# -------------------------------- 轨道偏离量随时间的变化 ----------------------------------
lt, lDr, lDphi, ltheta = simulation(n, a)

plt.plot(lt, lDr, c = 'red')
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
plt.xlabel(r'$t/T_s$')
plt.ylabel(r'$\Delta_r$/km')
plt.title(rf'$\alpha = {round(alpha * 180 / math.pi, 1)}$' + r'$^{\circ}, \Delta_r \sim t$')
plt.ion()
plt.pause(pause_time)
plt.close()

plt.plot(lt, lDphi, c = 'orange')
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
plt.xlabel(r'$t/T_s$')
plt.ylabel(r'$\Delta_{\phi}$')
plt.title(rf'$\alpha = {round(alpha * 180 / math.pi, 1)}$' + r'$^{\circ}, \Delta_{\phi} \sim t$')
plt.pause(pause_time)
plt.close()

plt.plot(lt, ltheta, c = 'deepskyblue')
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
plt.xlabel(r'$t/T_s$')
plt.ylabel(r'$\Theta$')
plt.title(rf'$\alpha = {round(alpha * 180 / math.pi, 1)}$' + r'$^{\circ}, \Theta \sim t$')
plt.pause(pause_time)
plt.close()

# ----------------------------------- 寻找最佳周期 --------------------------------------
la = np.linspace(1, 1.001, 11)
lmax = []
for a in la:
    lt, lDr, lDphi, ltheta = simulation(n, a)
    lmax.append(max(abs(x) for x in lDphi))
i = lmax.index(min(lmax))
print(f'偏差取极小值的a：{la[i]}')
print(f'最小偏差量：{lmax[i]}')
plt.plot(la, lmax, c = 'blue')
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
plt.xlabel(r'$a$')
plt.ylabel(r'$\max(|\Delta_{\phi}|)$')
plt.title(rf'$\alpha = {round(alpha * 180 / math.pi, 1)}$' + r'$^{\circ}, \max(|\Delta_{\phi}|) \sim a$')
plt.pause(pause_time)
plt.close()
plt.ioff()


time_end = time.time()          # 结束计时
print(f'运行结束，耗时{time_end - time_start}s')
