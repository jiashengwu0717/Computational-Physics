# 计算物理第一次大作业第1题源代码
# 运行该程序大约需要40s，运行过程中图像会自动关闭

pause_time = 5      # pause_time为每张图片的显示时间，默认为5s，可调

# 以下导入的除matplotlib外都是python内置库
import time
from copy import deepcopy
from timeit import Timer
from math import sqrt
import turtle
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Songti SC']

# 引用该矩阵类时索引从1开始取，即矩阵A的(i, j)元直接用A[i, j]取用
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
                import copy
                self.matrix[key - 1] = copy.deepcopy(value)   # 设置矩阵的第i行
            elif self.col == 1:
                self.matrix[key - 1][0] = value     # 设置列向量的第i个分量
        elif isinstance(key, tuple):
            self.matrix[key[0] - 1][key[1] - 1] = value

    def __add__(self, N):       # 矩阵的加法
        assert N.size == self.size, "维度不匹配，不能相加"
        M = Matrix((self.row, self.col))
        for r in range(self.row):
            for c in range(self.col):
                M[r, c] = self[r, c] + N[r, c]
        return M

    def __sub__(self, N):       # 矩阵的减法
        assert N.size == self.size, "维度不匹配，不能相减"
        M = Matrix((self.row, self.col))
        for r in range(self.row):
            for c in range(self.col):
                M[r, c] = self[r, c] - N[r, c]
        return M

    def norm(self, p=1):        # 范数，默认求列和范数
        if p == 1:          # 1-范数;列和范数
            norm = sum([abs(self.matrix[r][0]) for r in range(self.row)])
            for c in range(1, self.col):
                s = sum([abs(self.matrix[r][c]) for r in range(self.row)])
                if s > norm:
                    norm = s
        elif p == 0:         # inf-范数；行和范数
            norm = sum([abs(self.matrix[0][c]) for c in range(self.col)])
            for r in range(1, self.row):
                s = sum([abs(self.matrix[r][c]) for c in range(self.col)])
                if s > norm:
                    norm = s
        else:
            norm = None
        return norm

    def __str__(self, preference = 16):      # 直接输出的效果，每个矩阵元默认保留最大精度
        string = '['
        if self.row == 1 and self.col == 1:     # 只有一个矩阵元
            string += f'{round(self[1, 1], preference)} ]'
        elif self.row == 1 and self.col > 1:      # 只有一行
            for c in range(1, self.col):
                string += f'{str(round(self[1, c], preference))}  '
            string += f'{str(round(self[1, self.col], preference))} ]'
        elif self.row > 1 and self.col == 1:      # 只有一列
            string = '[ '
            for r in range(1, self.row):
                string += f'{str(round(self[r, 1], preference))}\n  '
            string += f'{str(round(self[self.row, 1], preference))} ]'
        else:
            for r in range(1, self.row):
                for c in range(1, self.col):
                    if r == 1 and c == 1:
                        string += '{:^3}'.format(str(round(self[r, c], preference)))
                    else:
                        string += '{:^4}'.format(str(round(self[r, c], preference)))
                string += '{:^4}\n'.format(str(round(self[r, self.col], preference)))
            for c in range(1, self.col):
                string += '{:^4}'.format(str(round(self[self.row, c], preference)))
            string += '{:^4}]'.format(str(round(self[self.row, self.col], preference)))
        return string




def solve(A:Matrix, b:Matrix):      # 针对该问题的线性方程组解法
    for k in range(1, A.row):
        for i in range(k + 1, min(k + 6, A.row + 1)):   # 对k行以下每一行进行消元，且只需搜到k+5
            A[i, k] /= - A[k, k]     # 倍乘因子
            for j in range(k + 1, A.col + 1):   # 扫描该行每个元素
                A[i, j] += A[i, k] * A[k, j]     # 更新矩阵元素
            b[i] += A[i, k] * b[k]
    for i in range(A.row, 0, -1):   # 回代法解方程
        for j in range(A.col, i, -1):
            b[i] -= A[i, j] * b[j]
        b[i] /= A[i, i]
    return b      # 返回解向量

def sor(A:Matrix, b:Matrix, x0:Matrix, M:int, e:float, w=0.89):      # successive over relaxation method
    x = Matrix((A.row, 1))
    k = 0
    while k < M:    # 最多迭代M次
        x[1] = (1 - w) * x0[1] + w * (b[1] - sum([A[1, j] * x0[j] for j in range(2, A.col+1)])) / A[1, 1]
        for i in range(2, A.row):
            x[i] = (1 - w) * x0[i] + w * (b[i] - sum([A[i, j] * x[j] for j in range(1, i)]) -
                                sum([A[i, j] * x0[j] for j in range(i + 1, A.col + 1)])) / A[i, i]
        x[A.row] = (1 - w) * x0[A.row] + w * (b[A.row] - sum([A[A.row, j] * x[j] for j in range(1, A.col)])) / A[A.row, A.col]
        if (x - x0).norm(0) < e:    # 已达到判停标准
            break
        x0 = deepcopy(x)       # 这样赋值必须deepcopy
        k += 1
    return x

def bridge_Matrix(l):       # 桥长l时待解方程组的系数矩阵
    A = Matrix((4*l, 4*l))
    A[1, 1] = -1
    A[2, 2] = -1
    for i in range(int(l/2)):
        A[4*i + 1, 4*i + 4] = 1/sqrt(2)
        A[4*i + 2, 4*i + 4] = 1/sqrt(2)
        A[4*i + 3, 4*i + 4] = -1/sqrt(2)
        A[4*i + 4, 4*i + 4] = -1/sqrt(2)
    for i in range(int(l/2), int(l) - 1):
        A[4*i - 1, 4*i + 6] = 1/sqrt(2)
        A[4*i, 4*i + 6] = -1/sqrt(2)
        A[4*i + 5, 4*i + 6] = -1/sqrt(2)
        A[4*i + 6, 4*i + 6] = 1/sqrt(2)
    A[4*l - 5, 4*l - 5] = 1/sqrt(2)
    A[4*l - 4, 4*l - 5] = -1/sqrt(2)
    A[4*l - 1, 4*l - 5] = -1/sqrt(2)
    A[4*l, 4*l - 5] = 1/sqrt(2)
    for i in range(l - 2):
        A[4*i + 3, 4*i + 3] = 1
        A[4*i + 7, 4*i + 3] = -1
    for i in range(l - 1):
        A[4*i + 1, 4*i + 5] = 1
        A[4*i + 5, 4*i + 5] = -1
    for i in range(int(l/2)):
        A[4*i + 4, 4*i + 6] = -1
        A[4*i + 6, 4*i + 6] = 1
    for i in range(int(l/2), int(l) - 1):
        A[4*i + 4, 4*i + 4] = -1
        A[4*i + 6, 4*i + 4] = 1
    A[4*l - 3, 4*l - 1] = 1
    A[4*l - 1, 4*l - 1] = -1
    A[4*l, 4*l] = -1
    return A

def draw(l:int, x:Matrix):          # 桁架中各杆受力示意图，红表示拉力，绿表示压力，色越鲜艳力越大
    d = 560/l       # 水平直杆长度
    r = 40/l        # 节点半径
    t = turtle.Pen()
    t.speed(0)
    t.hideturtle()          # 隐藏海龟
    t.pensize(int(24/l))    # 画笔粗细
    t.penup()
    t.backward(320)
    t.pendown()
    t.right(90)
    t.circle(r, 90)
    t.right(90)
    t.pencolor((x[2]/(2*x.norm(0))+0.5,0,0)) if x[2] > 0 else t.pencolor((0,-x[2]/(2*x.norm(0))+0.5,0))
    t.forward(d)
    t.penup()
    t.backward(d)
    t.pendown()
    t.left(90)
    t.pencolor('black')
    t.circle(r, 270)
    t.right(90)
    t.pencolor((x[1]/(2*x.norm(0))+0.5,0,0)) if x[1] > 0 else t.pencolor((0,-x[1]/(2*x.norm(0))+0.5,0))
    t.forward(d/2)
    t.penup()
    t.backward(d/2)
    t.pendown()
    t.left(180)
    t.penup()
    t.forward(2*r)
    t.pendown()
    for i in range(l):          # 开始画最下层桁架
        color = x[4*i+5] if (i != l - 1) else x[4*l-1]
        t.pencolor((color/(2*x.norm(0))+0.5,0,0)) if color > 0 else t.pencolor((0,-color/(2*x.norm(0))+0.5,0))
        t.forward(d)
        t.right(90)
        t.pencolor('black')
        t.circle(r)
        t.left(90)
        t.penup()
        t.forward(2*r)
        t.pendown()
    t.penup()
    t.left(90)
    t.circle(r, 270)
    t.right(90)
    t.pendown()
    t.pencolor((x[4*l]/(2*x.norm(0))+0.5,0,0)) if x[4*l] > 0 else t.pencolor((0,-x[4*l]/(2*x.norm(0))+0.5,0))
    t.forward(d)
    t.penup()
    t.backward(d+r)
    t.left(90)
    t.backward(l*(d+2*r) - r)
    t.left(90)
    t.circle(5,45)
    t.right(90)
    t.pendown()
    for i in range(int(l/2)):       # 中间层桁架-左半部分
        t.pencolor((x[4*i+4]/(2*x.norm(0))+0.5,0,0)) if x[4*i+4] > 0 else t.pencolor((0,-x[4*i+4]/(2*x.norm(0))+0.5,0))
        t.forward(sqrt(2)*(d+2*r) - 2*r)
        t.penup()
        t.right(90)
        t.circle(r, 45)
        t.right(90)
        t.pendown()
        t.pencolor((x[4*i+6]/(2*x.norm(0))+0.5,0,0)) if x[4*i+6] > 0 else t.pencolor((0,-x[4*i+6]/(2*x.norm(0))+0.5,0))
        t.forward(d)
        t.penup()
        if i != int(l/2) - 1:
            t.right(90)
            t.circle(r, 315)
            t.right(90)
            t.pendown()
    t.backward(d)
    t.left(90)
    t.circle(5,45)
    t.right(90)
    t.pendown()
    for i in range(int(l/2), l - 1):        # 中间层桁架-右半部分
        t.pencolor((x[4*i+6]/(2*x.norm(0))+0.5,0,0)) if x[4*i+6] > 0 else t.pencolor((0,-x[4*i+6]/(2*x.norm(0))+0.5,0))
        t.forward(sqrt(2)*(d+2*r) - 2*r)
        t.penup()
        t.right(90)
        t.circle(r, 315)
        t.right(90)
        t.pendown()
        t.pencolor((x[4*i+4]/(2*x.norm(0))+0.5,0,0)) if x[4*i+4] > 0 else t.pencolor((0,-x[4*i+4]/(2*x.norm(0))+0.5,0))
        t.forward(d)
        t.penup()
        t.right(90)
        t.circle(r, 45)
        t.right(90)
        t.pendown()
    t.pencolor((x[4*l-5]/(2*x.norm(0))+0.5,0,0)) if x[4*l-5] > 0 else t.pencolor((0,-x[4*l-5]/(2*x.norm(0))+0.5,0))
    t.forward(sqrt(2)*(d+2*r) - 2*r)
    t.penup()
    t.backward(sqrt(2)*(d+2*r) - r)
    t.left(45)
    t.backward((l-2)*(d+2*r) + r)
    t.right(90)
    t.pendown()
    t.pencolor('black')
    t.circle(r)
    for i in range(l - 2):      # 最上层桁架
        t.penup()
        t.left(90)
        t.forward(2*r)
        t.pendown()
        t.pencolor((x[4*i+3]/(2*x.norm(0))+0.5,0,0)) if x[4*i+3] > 0 else t.pencolor((0,-x[4*i+3]/(2*x.norm(0))+0.5,0))
        t.forward(d)
        t.right(90)
        t.pencolor('black')
        t.circle(r)
    turtle.Screen().bye()


time_start = time.time()        # 开始计时


# -----------------------------第二问-------------------------------
print('-----------------桥梁中各杆受力分布的求解------------------')
l = 8
A = bridge_Matrix(l)
print(f'l={l}时，方程组的系数矩阵为：')
print(A.__str__(1))
b = Matrix([[0 if i==2 or (i-6)%4 else 1] for i in range(1, 4*l + 1)])
x = solve(A, b)     # 解向量中各项即代表各杆受力情况
# 下面这部分只对于l = 8适用
lF = ['N1', 'N2', 'F2-4', 'F1-2', 'F1-3', 'F2-3', 'F4-6', 'F3-4', 'F3-5', 'F4-5', 'F6-8', 'F5-6', 'F5-7',
     'F6-7', 'F8-10', 'F7-8', 'F7-9', 'F8-9', 'F10-12', 'F10-11', 'F9-11', 'F8-11', 'F12-14', 'F12-13', 'F11-13',
     'F10-13', 'F14-16', 'F14-15', 'F13-15', 'F12-15', 'F15-16', 'N3']
print(('+-------')*16 + '+')
print('|',end = '')
print(f''.join('{:^7}|'.format(str(x)) for x in lF[:16]))
print(('+-------')*16 + '+')
print('|',end = '')
print(f''.join('{:^7}|'.format(str(round(x[i], 3))) for i in range(1, 17)))
print(('+-------')*16 + '+')
print('|',end = '')
print(f''.join('{:^7}|'.format(str(x)) for x in lF[16:32]))
print(('+-------')*16 + '+')
print('|',end = '')
print(f''.join('{:^7}|'.format(str(round(x[i], 3))) for i in range(17, 33)))
print(('+-------')*16 + '+')
print('')

draw(l, x)      # 画出受力示意图

# -----------------------------第三问-------------------------------
def maxForce_direction(l):         # 直杆所受最大力的直接解法
    A = bridge_Matrix(l)
    b = Matrix([[0 if i==2 or (i-6)%4 else 1] for i in range(1, 4*l + 1)])
    x = solve(A, b)
    force = round(x.norm(0), 10)    # 保留10位小数
    return force

def maxForce_iteration(l, M, e, w):     # 最大力迭代解法（桥长；最大迭代次数；判停标准；松弛因子）
    A = bridge_Matrix(l)
    b = Matrix([[0 if i==2 or (i-6)%4 else 1] for i in range(1, 4*l + 1)])
    x = sor(A, b, Matrix((A.row, 1, 1)), M, e, w)   # 从全是1的向量开始迭代
    force = round(x.norm(0), 10)    # 保留10位小数
    return force


print('---------------直杆所受最大力随桥梁长度的变化----------------')
ll = [i for i in range(8, 104, 2)]
lf = []
for l in ll:         # 1/8*l^2
    lf.append(maxForce_direction(l))   # lf存储了48个桥梁长度对应的对大力
print(('+-------')*17 + '+')
print('|   l   |',end = '')
print(f''.join('{:^7}|'.format(str(i)) for i in range(8, 40, 2)))
print(('+-------')*17 + '+')
print('| F-max |',end = '')
print(f''.join('{:^7}|'.format(str(x)) for x in lf[:16]))
print(('+-------')*17 + '+')
print('|   l   |',end = '')
print(f''.join('{:^7}|'.format(str(i)) for i in range(40, 72, 2)))
print(('+-------')*17 + '+')
print('| F-max |',end = '')
print(f''.join('{:^7}|'.format(str(x)) for x in lf[16:32]))
print(('+-------')*17 + '+')
print('|   l   |',end = '')
print(f''.join('{:^7}|'.format(str(i)) for i in range(72, 104, 2)))
print(('+-------')*17 + '+')
print('| F-max |',end = '')
print(f''.join('{:^7}|'.format(str(x)) for x in lf[32:48]))
print(('+-------')*17 + '+')
print('')

plt.plot(ll, lf, label = 'F-max - l')
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')      #设置上边和右边无边框
plt.xlabel('桥梁的长度')
plt.ylabel('直杆所受最大拉力')
plt.legend(loc='upper left', frameon=True)
plt.title(f'最大受力-桥梁长度关系图')
plt.ion()
plt.pause(pause_time)
plt.close()


print('-----------------直接法与迭代解法计算耗时比较--------------------')
l_direction, l_iteration = [], []
for l in range(8, 22, 2):
    t1 = Timer('maxForce_direction(l)','from __main__ import maxForce_direction, l')
    t2 = Timer('maxForce_iteration(l, 1000, 0.0001*(l**2), 0.89)','from __main__ import maxForce_iteration, l')
    l_direction.append(t1.timeit(number=1))
    l_iteration.append(t2.timeit(number=1))
print(('+----------')*8 + '+')
print('|  桥梁长度  |',end = '')
print(f''.join('{:^10}|'.format(str(i)) for i in range(8, 22, 2)))
print(('+----------')*8 + '+')
print('| 直接解法/s |',end = '')
print(f''.join('{:^10}|'.format(str(round(x, 6))) for x in l_direction))
print(('+----------')*8 + '+')
print('| 迭代解法/s |',end = '')
print(f''.join('{:^10}|'.format(str(round(x, 6))) for x in l_iteration))
print(('+----------')*8 + '+')
print('')


# -----------------------------第四问-------------------------------
print('-------------桥梁中直杆所受最大拉力随小汽车位置的变化---------------')
A = bridge_Matrix(8)
b = Matrix((32, 1))
f = []
for i in range(1, 8):
    b[4*i + 2] = 1
    f.append(solve(deepcopy(A), deepcopy(b)))     # 将各个独立解存入列表f中
    b[4*i + 2] = 0

for i in range(1, 9):
    print(f'当小汽车位于第{i}根杆上,距杆左端x处时，杆所受最大拉力为：',end = '')
    if i == 1:      # 小车位于第一根直杆上
        S = {(2*f[0][2]-2, 2+f[0][2]+f[1][2]+f[2][2]+f[3][2]+f[4][2]+f[5][2]+f[6][2])}    # 小车重力G对N_2有贡献
        j, jm = 3, 33
    elif i == 8:    # 小车位于最后一根直杆上
        S = {(-2*f[6][-1]+2, f[0][-1]+f[1][-1]+f[2][-1]+f[3][-1]+f[4][-1]+f[5][-1]+3*f[6][-1])}    # 小车重力G对N_3有贡献
        j, jm = 2, 32
    else:           # 小车位于中间的一根直杆
        S = {(2*f[i-1][2]-2*f[i-2][2], 2*f[i-2][2]+f[0][2]+f[1][2]+f[2][2]+f[3][2]+f[4][2]+f[5][2]+f[6][2])}
        j, jm = 3, 33
    while j < jm:
        if i == 1:
            t = (2*f[0][j], f[0][j]+f[1][j]+f[2][j]+f[3][j]+f[4][j]+f[5][j]+f[6][j])    # 第j根杆的受力
        elif i == 8:
            t = (-2*f[6][j], f[0][j]+f[1][j]+f[2][j]+f[3][j]+f[4][j]+f[5][j]+3*f[6][j])     # 第j根杆的受力
        else:
            t = (2*f[i-1][j]-2*f[i-2][j], 2*f[i-2][j]+f[0][j]+f[1][j]+f[2][j]+f[3][j]+f[4][j]+f[5][j]+f[6][j])
        for m in deepcopy(S):
            if m[1] >= t[1] and m[0] + m[1] >= t[0] + t[1]:
                break
            elif m[1] <= t[1] and m[0] + m[1] <= t[0] + t[1]:
                S.remove(m)
                S.add(t)
            else:
                S.add(t)
        j += 1
    if len(S) == 1:
        for m in S:
            print(f'{round(m[0],3)} * x + {round(m[1],3)}')
    else:
        string = ''
        for m in S:
            string += f'{round(m[0],3)} * x + {round(m[1],3)}, '
        string = string.rstrip(', ')
        print(f'max({string})')

x = [0, 2, 3, 3.143, 4, 4.857 ,5, 6, 8]
y = [8, 10, 11.25, 11.143, 12, 11.143, 11.25, 10, 8]
plt.plot(x, y, label = 'F - d')
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')      #设置上边和右边无边框
plt.xlabel('小车与桁架左端的距离')
plt.ylabel('直杆所受最大拉力')
plt.legend(loc='upper left', frameon=True)
plt.title(f'最大拉力-小车位置关系图')
plt.ion()
plt.pause(pause_time)
plt.close()
print('')


time_end = time.time()          # 结束计时
print(f'运行结束，耗时{time_end - time_start}s')

