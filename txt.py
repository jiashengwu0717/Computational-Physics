from math import sqrt
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Songti SC']
import numpy as np
from scipy.fftpack import fft

# ----------------------------------------------------------------
suml = lambda l: float(l[0]) * 3600 + float(l[1]) * 60 + float(l[2])        # 计算秒数

def draw_single(file:str):
    data = []
    for line in open(file, 'r'):     #设置文件对象并读取每一行文件
        data.append(line)
    lt = [suml(data[i].split()[1].split(':')) for i in range(2, len(data))]
    lt = [x - lt[0] for x in lt]
    ld = [float(data[i].split()[2]) for i in range(2, len(data))]
    plt.plot(lt, ld, label = 'I ～ t', color = 'deepskyblue')
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    plt.xlabel(r'$t$/s')
    plt.ylabel(r'$I_{(\Delta x)} - \overline{I_{(\Delta x)}}$')
    plt.legend(loc='upper right', frameon=True)
    plt.title(f'{file[32:-4]} I～t 图')
    plt.show()

def draw_dual(file:str, n=2):
    data = []
    for line in open(file, 'r'):     #设置文件对象并读取每一行文件
        data.append(line)
    lt = [suml(data[i].split()[1].split(':')) for i in range(1, len(data))]
    lt = [x - lt[0] for x in lt]
    ld1 = [float(data[i].split()[2]) for i in range(1, len(data))]
    ld2 = [float(data[i].split()[3]) for i in range(1, len(data))]
    plt.plot(lt, ld1, label = r'$I_x \sim t$', color = 'orange')
    if n == 2:
        plt.plot(lt, ld2, label = r'$I_{laser} \sim t$', color = 'deepskyblue')
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    plt.xlabel(r'$t$/s')
    plt.ylabel(r'$I_{(\Delta x)} - \overline{I_{(\Delta x)}}$')
    plt.legend(loc='upper right', frameon=True)
    plt.title(f'{file[32:-4]} I～t 图')
    plt.show()

sign = lambda x: 1 if x > 0 else -1    # 符号函数

def basic_func(x, lx:list, k):         # 拉格朗日插值基函数
    l = 1
    for t in lx:
        l *= 1 if t == lx[k] else (x - t) / (lx[k] - t)
    return l

def Lagrange(x, lx:list, ly:list):         # 拉格朗日插值多项式
    return sum([ly[k] * basic_func(x, lx, k) for k in range(len(lx))])

def Hermite(x, lx:list, ly:list, lm:list):          # 厄米插值多项式
    H = 0
    for j in range(len(lx)):
        alpha = (1 - 2*(x - lx[j])*sum([1/(lx[j]-lx[k]) for k in range(len(lx)) if k != j])) * basic_func(x, lx, j)**2
        beta = (x - lx[j]) * basic_func(x, lx, j)**2
        H += ly[j] * alpha + lm[j] * beta
    return H

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

def cubic_spline(lx:list, ly:list, conditions):         # 三次样条插值多项式
    if conditions[0] == 1:          # 给定边界一阶导
        a = [(lx[i+3] -lx[i+2]) / (lx[i+3] - lx[i+1]) for i in range(len(lx)-3)]
        b = [2 for i in range(len(lx)-2)]
        c = [(lx[i+1] - lx[i]) / (lx[i+2] - lx[i]) for i in range(len(lx)-3)]
        f = [(lx[2]-lx[1])/(lx[2]-lx[0]) * (3*(ly[1]-ly[0])/(lx[1]-lx[0])-conditions[1]) + 3*(lx[1]-lx[0])*(ly[2]-ly[1])/((lx[2]-lx[0])*(lx[2]-lx[1]))] \
            + [3*(a[i] * (ly[i+2]-ly[i+1])/(lx[i+2]-lx[i+1]) + c[i+1] * (ly[i+3]-ly[i+2])/(lx[i+3]-lx[i+2])) for i in range(len(lx)-4)] \
            + [3*(lx[-1]-lx[-2])*(ly[-2]-ly[-3])/((lx[-1]-lx[-3])*(lx[-2]-lx[-3])) + (lx[-2]-lx[-3])/(lx[-1]-lx[-3]) * (3*(ly[-1]-ly[-2])/(lx[-1]-lx[-2])-conditions[2])]
        lm = [conditions[1]] + tdm(a, b, c, f) + [conditions[2]]
    elif conditions[0] == 2:        # 给定边界二阶导
        a = [(lx[i+2] - lx[i+1]) / (lx[i+2] - lx[i]) for i in range(len(lx)-2)] + [1]
        b = [2 for i in range(len(lx))]
        c = [1] + [(lx[i+1] - lx[i]) / (lx[i+2] - lx[i]) for i in range(len(lx)-2)]
        f = [3*(ly[1]-ly[0])/(lx[1]-lx[0]) - (lx[1]-lx[0])/2*conditions[1]] \
            + [3*(a[i] * (ly[i+1]-ly[i])/(lx[i+1]-lx[i]) + c[i+1] * (ly[i+2]-ly[i+1])/(lx[i+2]-lx[i+1])) for i in range(len(lx)-2)] \
            + [3*(ly[-1]-ly[-2])/(lx[-1]-lx[-2]) + (lx[-1]-lx[-2])/2*conditions[2]]
        lm = tdm(a, b, c, f)
    else:
        lm = None
    return lm

def calculate(x, lx, ly, lm):
    for k in range(len(lx)):
        if lx[k] == x:
            return ly[k]
        elif lx[k] > x:
            return Hermite(x, [lx[k-1], lx[k]], [ly[k-1], ly[k]], [lm[k-1], lm[k]])

def golden_section(f, a, b, tol = 10**-5):     # 黄金分割一维优化
    t = (sqrt(5)-1) / 2
    x1 = a + (1-t)*(b-a)
    x2 = a + t*(b-a)
    while b - a > tol:
        if f(x1) > f(x2):
            a = x1
            x1 = x2
            x2 = a + t*(b-a)
        else:
            b = x2
            x2 = x1
            x1 = a + (1-t)*(b-a)
    return a

def extreme_value(lx:list, ly:list, n = 1):        # 极值点
    le = []                         # 储存极值点
    mean_y = sum(ly) / len(ly)      # 整体均值
    lm = cubic_spline(lx, ly, (2, 0, 0))        # 计算样条插值所需系数，采用自然边界条件
    change = sign(ly[0] - mean_y)               # 记录数据位于均值上方(1)/下方(-1)
    start = lx[0]                   # 第一个点（可能不是零点）
    for i in range(1, len(lx)):
        if (ly[i] - mean_y) * change <= 0:           # 越过零点
            stop = lx[i]
            if change < 0:      # 找极小值
                f = lambda x: calculate(x, lx, ly, lm)      # 插值函数
                point = golden_section(f, start, stop)      # 解出极值点
            else:               # 找极大值
                g = lambda x: -calculate(x, lx, ly, lm)     # 负插值函数
                point = golden_section(g, start, stop)      # 解出极值点
            if not le:          # 空列表，加入第一个极值点
                le.append(point)
            else:               # 一般，均匀插入几个点加入列表中
                d = (point - le[-1]) / n
                for _ in range(n):
                    le.append(le[-1] + d)
            start = stop            # 较大的零点记为新的start
            change *= -1            # 数据关于均值的大小改变
    return le

def draw_norm(file:str, n = 5):      # 标定后I-x分布
    data = []
    for line in open(file, 'r'):     # 设置文件对象并读取每一行文件
        data.append(line)
    lt = [suml(data[i].split()[1].split(':')) for i in range(1, len(data))]
    lt = [x - lt[0] for x in lt]                    # 均匀采样时刻
    lp1 = [float(data[i].split()[2]) for i in range(1, len(data))]      # 待标定 光功率
    lp2 = [float(data[i].split()[3]) for i in range(1, len(data))]      # 激光 光功率
    lte = extreme_value(lt, lp2, n)                 # 记下激光各峰、谷值对应的时刻t
    lx = [316.4 * i / 10**6 / n for i in range(len(lte))]               # 单位：mm
    lm = cubic_spline(lt, lp1, (2, 0, 0))
    lp = [calculate(lte[i], lt, lp1, lm) for i in range(len(lte))]     # 标定后 光功率
    plt.plot(lx, lp, label = f'{file[32:-4]}', color = 'red')
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    plt.xlabel(r'$x_{norm}$/mm')
    plt.ylabel(r'$I_{(\Delta x)}-\overline{I_{(\Delta x)}}$', rotation = 0)
    plt.legend(loc='upper right', frameon=True)
    plt.title(r'$(I-\overline{I}) \sim x_{norm}$')
    plt.show()
    return lp       # 返回光功率

def draw_spectrum(file:str, n=5, left=300, right=700):      # 光谱分布
    ly = draw_norm(file, n)        # 得到相对干涉光强分布
    x_min, x_max = 1, len(ly)
    for x in range(len(ly)-1, 0, -1):
        if 316.4 / n * len(ly) / x >= left:
            x_max = x           # 最大指标
            break
    for x in range(x_max, 0, -1):
        if 316.4 / n * len(ly) / x > right:
            x_min = x           # 最小指标
            break
    nx = 316.4 / n * len(ly) / np.arange(1, len(ly))       # 取后N-1项
    ny = abs(fft(ly)[1:])                                  # 取后N-1项
    nx = nx[x_min : x_max]
    ny = ny[x_min : x_max]
    plt.plot(nx, ny, label = f'{file[32:-4]}', color = 'gold')
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    plt.xlabel(r'$\lambda$/nm')
    plt.ylabel(r'$\mathcal{F}\{I-\overline{I}\}$', rotation = 0)
    plt.legend(loc='upper left', frameon=True)
    plt.title(r'$\mathcal{F}\{I_{(\Delta x)}-\overline{I_{(\Delta x)}}\}\ \sim \ \lambda$')
    plt.show()
    return nx, ny


# ----------------------------------------------------------------
'''
file_list = ['/Users/wujiasheng/Documents/wjs/LASER.txt',
             '/Users/wujiasheng/Documents/wjs/WHITE_LASER.txt',
             '/Users/wujiasheng/Documents/wjs/HG_LASER.txt',
             '/Users/wujiasheng/Documents/wjs/HG_YELLOW_LASER.txt']
# 实验数据txt文件

draw_single(file_list[0])
for i in [1, 2, 3]:
    draw_dual(file_list[i])
    draw_dual(file_list[i], 1)

draw_spectrum(file_list[1], 50, 200, 1200)      # 卤素灯白光光谱
draw_spectrum(file_list[2])                     # 汞灯光谱
draw_spectrum(file_list[3], 20)                 # 汞灯滤出黄光光谱
'''

file = '/Volumes/WJS/迈克尔孙干涉仪/迈克尔逊/wjs/WHITE_LASER.txt'
draw_dual(file)
lam_sampling, spec_30 = draw_spectrum(file, 50, 200, 1200)      # 卤素灯白光光谱

from scipy import interpolate
freq_sampling = (1 / lam_sampling)                              # 频率采样
spec_30 = spec_30
tck = interpolate.splrep(freq_sampling, spec_30, k = 3)                 # 导入样本点，生成参数
xx = np.linspace(min(freq_sampling), max(freq_sampling), 10000)         # 插值点
yy = interpolate.splev(xx, tck)                                         # 计算样条插值
y = np.fft.ifft(yy)
plt.plot(abs(y)[10:100])                                                # 傅立叶逆变换结果
plt.show()
