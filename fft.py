import math
import random
import scipy.fftpack as pack
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Songti SC']
plt.rcParams['axes.unicode_minus']=False

'''
def fft_2(x:list, w):             # 基本程序，以2为基底
    n = len(x)
    if n == 1:
        y = x
    else:
        y = []
        p = [x[2*i] for i in range(n//2)]       # 偶数子序列
        s = [x[2*i+1] for i in range(n//2)]     # 奇数子序列
        q = fft_2(p, w**2)              # 递归调用fft
        t = fft_2(s, w**2)
        for k in range(n):
            y.append(q[k % (n//2)] + w**k * t[k % (n//2)])
    return y

def fft_3(x:list, w):               # 以3为基底
    n = len(x)
    if n == 1:
        y = x
    else:
        y = []
        p = [x[3 * i] for i in range(n // 3)]
        s = [x[3 * i + 1] for i in range(n // 3)]
        u = [x[3 * i + 2] for i in range(n // 3)]
        q = fft_3(p, w ** 3)        # 递归调用fft
        t = fft_3(s, w ** 3)
        v = fft_3(u, w ** 3)
        for k in range(n):
            y.append(q[k % (n // 3)] + w ** k * t[k % (n // 3)] + w ** (2 * k) * v[k % (n // 3)])
    return y

def fft_p(x:list, w, p=2):        # 以质数p为基底，默认为2
    n = len(x)
    if n == 1:
        y = x
    else:
        y, lf = [], []
        for q in range(p):
            s = [x[p * i + q] for i in range(n // p)]
            t = fft_p(s, w ** p, p)
            lf.append(t)
        for k in range(n):
            y.append(sum(w ** (i * k) * lf[i][k % (n // p)] for i in range(p)))
    return y

def prime_factor(n:int):       # 正整数n的质因数分解
    lp = []
    i = 2
    while i <= n:
        if n % i == 0:
            lp.append(i)
            n //= i
            continue
        i += 1
    return lp
'''

def first_factor(n:int):        # 正整数n的第一个质因子
    i = 2
    while True:
        if n % i == 0:
            return i
        i += 1

def fft_factor(x:list, w):          # 使用质因数分解计算离散傅立叶变换
    n = len(x)
    if n == 1:
        y = x
    else:
        y, lf = [], []
        p = first_factor(n)         # 取出n的最小的质因子
        for q in range(p):
            s = [x[p * i + q] for i in range(n // p)]
            t = fft_factor(s, w ** p)           # 对每一段计算傅立叶变换
            lf.append(t)                        # 计算结果存入列表lf
        for k in range(n):
            y.append(sum(w ** (i * k) * lf[i][k % (n // p)] for i in range(p)))
    return y

def fft_wjs(x:list):                # 吴嘉晟定义的快速傅立叶变换
    w = math.e ** (-2j * math.pi / len(x))
    return fft_factor(x, w)

