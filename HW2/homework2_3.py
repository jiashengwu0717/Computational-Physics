# 计算物理第二次大作业第3题源代码
# 运行该程序大约需要60s以上
# 运行过程中图像会自动关闭

pause_time = 5      # pause_time为每张图片的显示时间，默认为5s，可调

import time
import random                       # 产生随机数
from math import log10
from scipy.special import kv        # 用于计算第二类改良贝塞尔函数
import numpy as np                  # 方便计算
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Songti SC']
plt.rcParams['axes.unicode_minus']=False


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

def Integrate(f, b, lp, lc, n=20):        # 针对本题，计算函数在b—>+Infty上的积分，20个高斯点算得非常准确
    a = 0.5       # 取a+2b为中点
    return sum(lc[i] * 2*(a+b)/(1-lp[i])**2 * f((a+2*b+a*lp[i]) / (1-lp[i])) for i in range(n))


time_start = time.time()      # 开始计时

# ----------------------- 第一问：查表程序 ------------------------
def find(X, P):             # 查表程序
    Xi1 = int(99 / 5 * (log10(X) + 3))      # 较小X的索引指标
    llw = []                                # 储存log(w1')和log(w2')
    with open('c0.table', 'r') as f:
        for line in f.readlines()[Xi1 : Xi1 + 2]:   # 取出X1，X2对应的行
            l = line.split()
            start, end = 0, 499                 # 两个w的索引指标
            while end - start > 1:              # 直到确定两个相邻点为止
                mid = (start + end) // 2        # 二分查找
                if float(l[mid]) <= P:
                    start = mid
                else:
                    end = mid
            llw.append((-4 + 4 / 499 * start) * (log10(float(l[end])) - log10(P)) / (log10(float(l[end])) - log10(float(l[start]))) \
                  + (-4 + 4 / 499 * end) * (log10(P) - log10(float(l[start]))) / (log10(float(l[end])) - log10(float(l[start]))))
    log_w = llw[0] * (99 / 5 * (-3 + 5/99*(Xi1+1) - log10(X))) + llw[1] * (99 / 5 * (log10(X) - (-3 + 5/99*Xi1)))
    return 10 ** log_w

# ------------------- 第二问：辐射光子数关于光子能量的曲线 -------------------
aw = np.linspace(0.005, 0.995, 100)
aN = np.zeros(100)
for _ in range(100000):         # 10万次实验
    w = find(1, random.uniform(0, 1))
    aN[int(100 * w)] += 1/1000
plt.plot(aw, aN, label = r'$\dfrac{dN}{d\omega} - \omega$', marker = '*', markersize = 4)
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')      #设置上边和右边无边框
plt.xlabel(r'$\omega$')
plt.ylabel(r'$\dfrac{dN}{d\omega}$', rotation = 0)
plt.legend(loc='upper right', frameon=True)
plt.title(r'$\dfrac{dN}{d\omega} \sim \omega$关系图')
plt.ion()
plt.pause(pause_time)
plt.close()

# ------------------------- 理论曲线 ------------------------
X = 1
w = np.linspace(0.0001, 0.9999, 1000)       # 光子的相对能量
lp, lc = Gauss_coefficient(20)              # 取20个高斯点
W = (2 + w**2/(1-w)) * kv(2/3, 2*w/(3*X*(1-w))) - Integrate(lambda x:kv(1/3, x), 2*w/(3*X*(1-w)), lp, lc)   # 电子同步辐射谱
W = W / sum(W) * 1000             # 归一化
plt.plot(w, W, label = r'$\dfrac{dW}{dt d\omega} - \omega$', c = 'orange')
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')      #设置上边和右边无边框
plt.xlabel(r'$\omega$')
plt.ylabel(r'$\dfrac{dW}{dt d\omega}$',labelpad=20,rotation = 0)
plt.legend(bbox_to_anchor=(0.88, 1), loc='upper right', frameon=True)
plt.title(r'$\dfrac{dW}{dt d\omega} \sim \omega$关系图')
plt.ion()
plt.pause(pause_time)
plt.close()

# -------------------模拟结果与理论结果对比---------------------
plt.plot(aw, aN, label = '模拟统计曲线', c = 'red', marker = 'o', markersize = 3.5)
plt.plot(w, W, label = '理论计算曲线', c = 'yellow', ls = '--')
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
plt.xlabel(r'$\omega$')
plt.ylabel(r'$\dfrac{dN}{d\omega}$', rotation = 0)
plt.legend(bbox_to_anchor=(0.9, 0.9), loc='upper right', frameon=True)
plt.title('光子辐射能量概率密度分布图')
plt.ion()
plt.pause(pause_time)
plt.close()

# --------------------- 对数坐标下的理论结果 --------------------------
w = np.linspace(0.0001, 0.96, 1000)       # 光子的相对能量
W = (2 + w**2/(1-w)) * kv(2/3, 2*w/(3*X*(1-w))) - Integrate(lambda x:kv(1/3, x), 2*w/(3*X*(1-w)), lp, lc)   # 电子同步辐射谱
W = W / sum(W) * 1000             # 归一化
plt.plot(w, np.log10(W), label = r'$\lg(\dfrac{dW}{dt d\omega}) - \omega$', c = 'orange')
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')      #设置上边和右边无边框
plt.xlabel(r'$\omega$')
plt.ylabel(r'$\lg(\dfrac{dW}{dt d\omega})$',labelpad=20,rotation = 0)
plt.legend(bbox_to_anchor=(0.88, 1), loc='upper right', frameon=True)
plt.title(r'$\lg(\dfrac{dW}{dt d\omega}) \sim \omega$关系图')
plt.ion()
plt.pause(pause_time)
plt.close()


time_end = time.time()      # 结束计时
print(f'运行结束，耗时{time_end - time_start}s')
