from copy import deepcopy
from math import sin, cos, sqrt
from functools import reduce


class Vector:                           # 向量类
    def __init__(self, value):
        if isinstance(value, tuple):    # 已知维度 创建向量
            self.size = value[0]
            if len(value) > 1:
                self.vec = [value[1] for _ in range(value[0])]
            else:
                self.vec = [0 for _ in range(value[0])]
        elif isinstance(value, list):   # 已知各分量 创建向量
            self.size = len(value)
            self.vec = value

    def __getitem__(self, item):            # 按索引取值
        return self.vec[item - 1]           # 索引从1开始

    def __setitem__(self, key, value):      # 给某分量赋值
        self.vec[key - 1] = value

    def __add__(self, v):           # 加法运算
        assert v.size == self.size, '维度不匹配，不能相加'
        return Vector([self[i] + v[i] for i in range(1, self.size+1)])

    def __sub__(self, v):           # 减法运算
        assert v.size == self.size, '维度不匹配，不能相减'
        return Vector([self[i] - v[i] for i in range(1, self.size+1)])

    def __mul__(self, t):           # 数乘
        return Vector([x * t for x in self.vec])

    def __truediv__(self, t):       # 数除
        return Vector([x / t for x in self.vec])

    def __str__(self):              # 输出效果：以元组形式打印各分量
        return str(tuple(self.vec))



def sign(x):        # 自定义符号函数
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0

def diff(f, x, n=1):        # 差商型求导
    h = 10 ** (n-6)
    if n == 1:
        return (f(x+h) - f(x-h)) / (2*h)
    else:
        return (diff(f, x+h, n-1) - diff(f, x-h, n-1)) / (2*h)

def Bisection(f, a, b):     # 二分法
    while (b - a) / 2 > (10 ** -10):        # tol 可改
        c = a + (b - a)/2
        if sign(f(a)) == sign(f(c)):
            a = c
        else:
            b = c
    return a

print('-------------二分法求根--------------')
f = lambda x: x**2 - 4*sin(x)
print(Bisection(f, 1, 3))


def Newton_Method(x0, f):       # 牛顿法
    x = x0
    while abs(f(x0)) > (10 ** -10) or abs(x0 - x) > (10 ** -10):
        x = x0 - f(x0) / diff(f,x0)     # x存储的是x_{k+1}
        x0, x = x, x0       # 交换后，x0代表迭代得到的最新结果，x代表上一部迭代结果
    return x0

print('------------牛顿法求根-------------')
f = lambda x: cos(2*x)**2 - x**2
print(Newton_Method(1, f))


def Brent(a, b, f, e1, e2, delta):       # Brent算法
    if sign(f(a)) * sign(f(b)) >= 0:
        assert '无根或多根'
        return
    if abs(f(a)) <  abs(f(b)):
        a, b = b, a
    c = a
    mflag = 1
    s, d = 0, 0
    while abs(f(b)) > e1 or abs(b - a) > e2:
        if f(a) != f(c) and f(b) != f(c):       # 反内插
            s = (a * f(b) * f(c)) / ((f(a)-f(b)) * (f(a)-f(c))) + (b * f(a) * f(c)) / ((f(b)-f(a)) * (f(b)-f(c))) \
                + (c * f(a) * f(b)) / ((f(c)-f(a)) * (f(c)-f(b)))
        else:       # 割线法
            s = b - f(b) * (b - a) / (f(b) - f(a))
        if (s < (3*a+b)/4 or s > b) or (mflag == 1 and abs(s-b) > abs(b-c)/2) or (mflag == 0 and abs(s-b) > abs(c-d)/2) \
                or (mflag == 1 and abs(b-c) < delta) or (mflag == 0 and abs(c-d) < delta):      # 二分法
            s = (a + b) / 2
            mflag = 1
        else:
            mflag = 0
        d, c = c, b
        if sign(f(a)) * sign(f(s)) < 0:
            b = s
        else:
            a = s
        if abs(f(a)) <  abs(f(b)):
            a, b = b, a
    return b

print('-----------Brent算法-----------')
f = lambda x: (x + 3) * (x - 1)**2
print(Brent(-4, 4/3, f, 10**-10, 10**-10, 10**-10))


def wbrf(a, b, f):      # World's Best Root Finder
    x = a
    while f(x) < 10 ** -10:
        c = (a + b) / 2
        if sign(f(a)) * sign(f(c)) > 0:
            a, b = b, a
        if (f(b) * (f(b) - f(a)) - 2 * f(c) * (f(c) - f(a))) < 0:
            b, c = c, b
            continue
        B = (c - a) / (f(c) - f(a))     # f(c) = f(a)如何处理？
        C = (f(c) - f(a) - f(b)) / ((f(b) - f(a)) * (f(b) - f(c)))
        x = a - B * f(a) * (1 - C * f(c))
        if sign(f(x)) * sign(f(a)) < 0:
            b = x
        else:
            a, b = x, c
    return x

print('-----------世界上最好?的求根算法-----------')
f = lambda x: (x + 3) * (x - 1)**2
print(wbrf(-4, -2.5, f))


def Horner(a:list, x0, e, Nm):      # 秦九韶方法
    z = x0       # x_k
    b = deepcopy(a)
    c = deepcopy(b)
    k = 0
    while k < Nm:
        for j in range(len(a) - 2, 0, -1):
            b[j] = a[j] + z * b[j + 1]
            c[j] = b[j] + z * c[j + 1]
        b[0] = a[0] + z * b[1]
        z0, z = z, z - b[0]/c[1]
        if abs(z - z0) < e:
            break
    return z

print('-----------秦九韶方法求解多项式的根-----------')
a = [6, 20, 5, -40, 16]     # f(x) = 16x^4 - 40 x^3 + 5x^2 + 20x + 6
print(Horner(a, 1, 10**-8, 1000))


def golden_section(f, a, b, tol = 10**-10):     # 黄金分割一维优化
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

print('-------------黄金分割搜索算法------------')
f = lambda x: x**3 - x
print(golden_section(f, 0, 1))


def deviation(f, l):        # 判停标准的偏差函数
    mean = sum(f(x) for x in l) / (len(l))     # f的平均值
    delta_square = sum((f(x) - mean)**2 for x in l) / len(l)    # 方差
    return delta_square

def Nelder_Mead(f, lx:list, M=100, e = 10**-20, alpha=1, gamma=2, rho=1/2, sigma=1/2):     # NM simplex algorithm method
    n = len(lx) - 1             # lx储存所有试验点（n维向量）
    s = 0                       # 记录迭代次数
    while s < M and deviation(f, lx) > e:       # 迭代次数达到上限或精度达到要求即终止
        lx.sort(key=lambda x: f(x))                     # 按函数值从小到大排序，sort()使用快速排序法
        xo = reduce(lambda x, y: x+y, lx[:-1]) / n      # 计算除掉x_{n+1}点外所有n个点的重心
        xr = xo + (xo - lx[n]) * alpha                  # 计算反射点
        if f(xr) < f(lx[n-1]) and f(xr) >= f(lx[0]):    # x_r的函数值处于x_1和x_n之间
            lx[n] = xr                                  # 用x_r代替x_{n+1}，得到新的单纯形
        elif f(xr) < f(lx[0]):                  # x_r是最好的
            xe = xo + (xr - xo) * gamma             # 计算扩展点
            if f(xe) < f(xr):                       # x_e优于x_r
                lx[n] = xe                          # 用x_e代替x_{n+1}，得到新的单纯形
            else:                                   # 还是x_r优于x_e
                lx[n] = xr                          # 用x_r代替x_{n+1}，得到新的单纯形
        else:                                   # 反射点还不如次差点
            xc = xo + (lx[n] - xo) * rho            # 计算收缩点
            if f(xc) < f(lx[n]):                    # x_c优于x_{n+1]
                lx[n] = xc                          # x_c代替x_{n+1]
            else:                               # 收缩点还不如最差点
                for i in range(1, n+1):
                    lx[i] = lx[0] + (lx[i] - lx[0]) * sigma         # 折减
        s += 1
        print(f'{s}， 最优点：{lx[0]}， 方差：{deviation(f, lx)}')
    return lx[0]                # 将x_1作为最优点返回

print('--------------------单纯形法---------------------')
def f(v:Vector):            # 向量函数，最小值点(2,-2)
    x, y = v.vec
    return 3/2 * x**2 + 3 * y**2 + 2*x*y - 2*x + 8*y
ld = [Vector([10, -13]), Vector([-10, 2]), Vector([17.5, -3])]
point = Nelder_Mead(f, ld, 100)







