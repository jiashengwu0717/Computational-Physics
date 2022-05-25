import math


def basic_func(x, lx:list, k):         # 拉格朗日插值基函数
    l = 1
    for t in lx:
        l *= 1 if t == lx[k] else (x - t) / (lx[k] - t)
    return l

def Lagrange(x, lx:list, ly:list):         # 拉格朗日插值多项式
    return sum([ly[k] * basic_func(x, lx, k) for k in range(len(lx))])


print('-------拉格朗日插值-------')
# import sympy
# x = sympy.Symbol('x')
lx = [-2, -1, 0, 1, 2]
ly = [0.25, 0.5, 1, 2, 4]
print(Lagrange(0.3, lx, ly))


def Hermite(x, lx:list, ly:list, lm:list):          # 厄米插值多项式
    H = 0
    for j in range(len(lx)):
        alpha = (1 - 2*(x - lx[j])*sum([1/(lx[j]-lx[k]) for k in range(len(lx)) if k != j])) * basic_func(x, lx, j)**2
        beta = (x - lx[j]) * basic_func(x, lx, j)**2
        H += ly[j] * alpha + lm[j] * beta
    return H

print('--------厄米插值--------')
lx = [-1, 0, 1, 2]
ly = [1.937, 1, 1.349, -0.995]
lm = [-1.635, 0.707, -1.526, -0.706]
print(Hermite(0, lx, ly, lm))


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

print('--------三对角矩阵的追赶法-------')
a,b,c,f = [3, -2], [1, 2, -1] , [-4, 5], [1, 8 ,-1]
print(tdm(a, b, c, f))      # x为list类型


def cubic_spline(x, lx:list, ly:list, conditions):         # 三次样条插值多项式
    if conditions[0] == 1:
        a = [(lx[i+3] -lx[i+2]) / (lx[i+3] - lx[i+1]) for i in range(len(lx)-3)]
        b = [2 for i in range(len(lx)-2)]
        c = [(lx[i+1] - lx[i]) / (lx[i+2] - lx[i]) for i in range(len(lx)-3)]
        f = [(lx[2]-lx[1])/(lx[2]-lx[0]) * (3*(ly[1]-ly[0])/(lx[1]-lx[0])-conditions[1]) + 3*(lx[1]-lx[0])*(ly[2]-ly[1])/((lx[2]-lx[0])*(lx[2]-lx[1]))] \
            + [3*(a[i] * (ly[i+2]-ly[i+1])/(lx[i+2]-lx[i+1]) + c[i+1] * (ly[i+3]-ly[i+2])/(lx[i+3]-lx[i+2])) for i in range(len(lx)-4)] \
            + [3*(lx[-1]-lx[-2])*(ly[-2]-ly[-3])/((lx[-1]-lx[-3])*(lx[-2]-lx[-3])) + (lx[-2]-lx[-3])/(lx[-1]-lx[-3]) * (3*(ly[-1]-ly[-2])/(lx[-1]-lx[-2])-conditions[2])]
        lm = [conditions[1]] + tdm(a, b, c, f) + [conditions[2]]

    elif conditions[0] == 2:
        a = [(lx[i+2] - lx[i+1]) / (lx[i+2] - lx[i]) for i in range(len(lx)-2)] + [1]
        b = [2 for i in range(len(lx))]
        c = [1] + [(lx[i+1] - lx[i]) / (lx[i+2] - lx[i]) for i in range(len(lx)-2)]
        f = [3*(ly[1]-ly[0])/(lx[1]-lx[0]) - (lx[1]-lx[0])/2*conditions[1]] \
            + [3*(a[i] * (ly[i+1]-ly[i])/(lx[i+1]-lx[i]) + c[i+1] * (ly[i+2]-ly[i+1])/(lx[i+2]-lx[i+1])) for i in range(len(lx)-2)] \
            + [3*(ly[-1]-ly[-2])/(lx[-1]-lx[-2]) + (lx[-1]-lx[-2])/2*conditions[2]]
        lm = tdm(a, b, c, f)

    else:
        lm = None

    for k in range(len(lx) + 1):
        if lx[k] == x:
            return ly[k]
        elif lx[k] > x:
            return Hermite(x, [lx[k-1], lx[k]], [ly[k-1], ly[k]], [lm[k-1], lm[k]])





print('-------三次样条插值-------')
lx = [0, 1, 2, 3, 4]
ly = [0, 1, 0, -1, 0]
print(cubic_spline(3.2, lx, ly, (2, 0, 0)))
print(cubic_spline(3.2, lx, ly, (1, 3/2, 3/2)))














