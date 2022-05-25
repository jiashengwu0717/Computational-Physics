from math import sin, sqrt


def diff(f, x, n=1):        # 默认求一阶导数
    h = 10 ** (n-6)
    if n == 1:
        return (f(x+h) - f(x-h)) / (2*h)
    else:
        return (diff(f, x+h, n-1) - diff(f, x-h, n-1)) / (2*h)

print('--------------差商型求导--------------')
f = lambda x: x ** 7
print(diff(f, 1, 1))
print(diff(f, 1, 2))
print(diff(f, 1, 3))
print(diff(f, 1, 4))
print(diff(f, 1, 5))
print(diff(f, 1, 6))


def complex_trapezoidal(f, a, b, n):    # 复化梯形积分公式
    h = (b - a) / n
    x = [f(a + i*h) for i in range(1, n)]
    T = h / 2 * (f(a) + 2 * sum(x) + f(b))
    return T

def complex_quadratic(f, a, b, n):      # 复化抛物线积分公式
    h = (b - a) / n
    x1 = [f(a + (i+1/2)*h) for i in range(n)]
    x2 = [f(a + i*h) for i in range(1, n)]
    S = h / 6 * (f(a) + 4*sum(x1) + 2*sum(x2) + f(b))
    return S

def complex_cotes(f, a, b, n):          # 复化柯特斯积分公式
    h = (b - a) / n
    x1 = [f(a + (i+1/4)*h) for i in range(n)]
    x2 = [f(a + (i+1/2)*h) for i in range(n)]
    x3 = [f(a + (i+3/4)*h) for i in range(n)]
    x4 = [f(a + i*h) for i in range(1, n)]
    C = h / 90 * (7*f(a) + 32*sum(x1) + 12*sum(x2) + 32*sum(x3) + 14*sum(x4) + 7*f(b))
    return C

print('--------------复化求积法---------------')
f = lambda x: sin(x)/x if x != 0 else 1
print(complex_trapezoidal(f, 0, 1, 8))
print(complex_quadratic(f, 0, 1, 4))
print(complex_cotes(f, 0, 1, 2))
print('')
f = lambda x: 1 / (x + 1)**2
print(complex_trapezoidal(f, 1, 3, 8))
print(complex_quadratic(f, 1, 3, 8))
print(complex_cotes(f, 1, 3, 8))



def sign(x):        # 自定义符号函数
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0

def Legendre(l, x):         # 勒让德多项式，l=0,1,2,···
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

def Newton_Method(f, x0):       # 牛顿法找x0附近的零点
    x = x0
    while abs(f(x0)) > (10 ** -10) or abs(x0 - x) > (10 ** -10):
        x = x0 - f(x0) / diff(f,x0)     # x存储的是x_{k+1}
        x0, x = x, x0       # 交换后，x0代表迭代得到的最新结果，x代表上一部迭代结果
    return x0

def Gauss_point(n):         # 生成高斯点列表
    f = lambda x: Legendre(n, x)        # 给定n，生成n阶勒让德多项式
    lp = [Newton_Method(f, -1)]
    for _ in range(n-1):
        def g(x):
            if x in lp:
                g_value = Legendre_diff(n, x)
                for t in lp:
                    g_value /= (x - t) if t != x else 1     # 除掉已得到的因子
            else:
                g_value = Legendre(n, x)
                for t in lp:
                    g_value /= (x - t)      # 除掉已得到的因子
            return g_value
        lp.append(Newton_Method(g, -1))
    return lp

def basic_func(lx:list, k, x):         # 拉格朗日插值基函数
    l = 1
    for t in lx:
        l *= 1 if t == lx[k] else (x - t) / (lx[k] - t)
    return l

def Gauss_coefficient(n):
    f = lambda x: Legendre(n, x)        # 给定n，生成n阶勒让德多项式
    lp = [Newton_Method(f, -1)]         # lp存储n个高斯点
    for _ in range(n-1):
        def g(x):
            if x in lp:
                g_value = Legendre_diff(n, x)
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
        C = complex_cotes(h, -1, 1, 4)      # 使用4阶cotes公式计算积分值
        lc.append(C)
    return lp, lc



print('-------------数值求解高斯点-------------')
print(Gauss_coefficient(17)[0])

print('--------------计算求积系数--------------')
print(Gauss_coefficient(17)[1])

def integrate_Gauss_Legendre(f, a, b, n):     # 具有2n+1次代数精度的Gauss-Legendre积分公式
    g = lambda x: f((b-a)/2 * x + (a+b)/2)      # 做积分区间变换
    lp, lc = Gauss_coefficient(n)
    return (b-a)/2 * sum([lc[i] * g(lp[i]) for i in range(n)])

print('--------------高斯-勒让德积分--------------')
f = lambda x: sin(x)/x if x != 0 else 1
n = 10
s = integrate_Gauss_Legendre(f, 0, 1, n)
for i in range(1, 100):
    s += integrate_Gauss_Legendre(f, i, i+1, n)
print(s)
print(integrate_Gauss_Legendre(f, 0, 100, n))

print('----------------------------------')
a = 1
g = lambda y: f(a*(1+y)/(1-y)) / (1-y)**2
print(2*a*integrate_Gauss_Legendre(g, -1, 1, 10))



print('--------------test--------------')
a = 10
f = lambda x: 1 / (1 + x**2)
g = lambda y: 2*a * f(a*(1+y)/(1-y)) / (1-y)**2
print(integrate_Gauss_Legendre(g, -1, 1, 10))








