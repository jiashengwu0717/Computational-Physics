# 计算物理第二次大作业第1题源代码
# 运行该程序时，需将'matrix.py'文件和'eigen.py'文件放在该文件的同一目录下
# 运行该程序大约需要100s以上
# 运行过程中图像会自动关闭

pause_time = 5      # pause_time为每张图片的显示时间，默认为5s，可调

import time
import math
from math import sqrt, sin, cos
from matrix import Matrix, I_Matrix                     # 矩阵类
from eigen import QR_Wilkinson, QRD, power_method_min   # 将用到的求解本征值问题的方法
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Songti SC']
plt.rcParams['axes.unicode_minus']=False


class generate_A(Matrix):       # 生成本题的矩阵A
    def __init__(self, N, t=1.0):
        matrix = [[-2 if r == c else 0 for c in range(N)] for r in range(N)]
        matrix[0][N - 1], matrix[N - 1][0] = 1, t
        for i in range(N - 1):
            if i % 2:       # 奇数
                matrix[i][i + 1], matrix[i + 1][i] = t, 1
            else:           # 偶数
                matrix[i][i + 1], matrix[i + 1][i] = 1, t
                matrix[i + 1][i + 1] *= t
        super(generate_A, self).__init__(matrix)

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

def dispertion_num(N):      # 一维单原子链色散关系的数值计算
    Eig, Q = QR_Wilkinson(generate_A(N))
    lk1 = [2 * math.pi / N * l for l in range(N // 2 + 1)]
    lw1 = [0.0 for _ in range(N // 2 + 1)]
    for i in range(1, N+1):
        line = Q.getCol_list(i)                             # 取出变换矩阵的第i列
        l = [abs(x) for x in fft_wjs(line)[: N // 2 + 1]]   # 对傅立叶变换结果取模，由对称性只需取变换后的一半数据
        lw1[l.index(max(l))] = sqrt(abs(Eig[i - 1]))        # 将omega_k加入与k相对应的位置
    lk2 = [-x for x in lk1[1:]]  # 对称性
    lw2 = lw1[1:]  # 对称性
    plt.scatter(lk1, lw1, label='数值结果', c = 'red', s = 10)
    plt.scatter(lk2, lw2, c='red', s = 10)
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    plt.xlabel(r'$k$')
    plt.ylabel(r'$\omega$', rotation=0)
    plt.legend(bbox_to_anchor=(1, 0.5), loc='upper right', frameon=True)
    plt.title(f'N = {N} 色散关系')
    plt.ion()
    plt.pause(pause_time)
    plt.close()

def dispertion_thm(N):      # 一维单原子链色散关系的解析计算
    lk1 = [2 * math.pi / N * l for l in range(N // 2 + 1)]
    lw1 = [2 * sin(k/2) for k in lk1]
    lk2 = [-x for x in lk1[1:]]         # 对称性
    lw2 = lw1[1:]                       # 对称性
    plt.plot(lk1, lw1, label = '解析结果', c = 'orange')
    plt.plot(lk2, lw2, c = 'orange')
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    plt.xlabel(r'$k$')
    plt.ylabel(r'$\omega$', rotation=0)
    plt.legend(bbox_to_anchor=(1, 0.5), loc='upper right', frameon=True)
    plt.title(f'色散关系解析计算结果')
    plt.ion()
    plt.pause(pause_time)
    plt.close()

def dispertion_1(N):        # 一维单原子链色散关系数值结果与解析结果对比
    Eig, Q = QR_Wilkinson(generate_A(N))
    lk = [2 * math.pi / N * l for l in range(N // 2 + 1)]
    lwn = [0.0 for _ in range(N // 2 + 1)]
    for i in range(1, N + 1):
        line = Q.getCol_list(i)  # 取出变换矩阵的第i列
        l = [abs(x) for x in fft_wjs(line)[: N // 2 + 1]]  # 对傅立叶变换结果取模，由对称性只需取变换后的一半数据
        lwn[l.index(max(l))] = sqrt(abs(Eig[i - 1]))  # 将omega_k加入与k相对应的位置
    lk = [-x for x in reversed(lk[1:])] + lk
    lwn = [x for x in reversed(lwn[1:])] + lwn
    lwt = [2 * abs(sin(k/2)) for k in lk]
    plt.scatter(lk, lwn, label = '数值结果', c='red', s=9)
    plt.plot(lk, lwt, label = '解析结果', c = 'orange', ls = '--')
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    plt.xlabel(r'$k$')
    plt.ylabel(r'$\omega$', rotation=0)
    plt.legend(bbox_to_anchor=(1, 0.5), loc='upper right', frameon=True)
    plt.title(f'N = {N}；单原子链色散关系曲线')
    plt.ion()
    plt.pause(pause_time)
    plt.close()

def dispertion_2(N, t):         # 一维双原子链色散关系对比
    T, Q = QRD(generate_A(N, t))
    lk = [4 * math.pi / N * l for l in range(N // 2 + 1)]
    lw = [0.0 for _ in range(N // 2 + 1)]
    for i in range(1, N + 1):
        vec = Q.getCol(i)       # 取出变换矩阵的第i列列矢量
        lam, line = power_method_min(generate_A(N, t) - I_Matrix(N) * (T[i, i] + 10 ** -5), vec)  # 引入10**-5让算法更稳定
        lam += T[i, i] + 10 ** -5   # 修正本征值
        line = [x for x in line]    # 将本征矢改为列表类型
        l = [abs(x) for x in fft_wjs(line)[: N // 2 + 1]]  # 对傅立叶变换结果取模，由对称性只需取变换后的一半数据
        index = l.index(max(l))  # 非零元素的指标
        if lw[index] == 0:
            lw[index] = sqrt(abs(lam))  # 将omega_k加入与k相对应的位
        else:
            if abs(lw[index] - sqrt(abs(lam))) > 0.01:  # 同一个k对应不同的omega
                lk.append(index)  # 先将指标存在lk末尾
                lw.append(sqrt(abs(lam)))  # 将本征值存在lw末尾
    if len(lk) > N // 2 + 1:  # 出现同一个k对应不同的omega的情况
        index = int(lk.pop())
        if lw[index] > lw[-1]:
            lk.insert(index, 4 * math.pi / N * index)
            lw.insert(index, lw.pop())
        else:
            lk.insert(index + 1, 4 * math.pi / N * index)
            lw.insert(index + 1, lw.pop())
    lk1 = [-x for x in reversed(lk[1: N // 4 + 1])] + lk[: N // 4 + 1]
    lw1n = [x for x in reversed(lw[1: N // 4 + 1])] + lw[: N // 4 + 1]
    lw1t = [sqrt(t + 1 - sqrt(t ** 2 + 1 + 2 * t * cos(k))) for k in lk1]
    lk2 = [x - 2 * math.pi for x in lk[N // 4 + 1:]] + [2 * math.pi - x for x in reversed(lk[N // 4 + 1: -1])]
    lw2n = lw[N // 4 + 1:] + [x for x in reversed(lw[N // 4 + 1: -1])]
    lw2t = [sqrt(t + 1 + sqrt(t ** 2 + 1 + 2 * t * cos(k))) for k in lk2]
    plt.scatter(lk1, lw1n, label='数值结果', c='red', s=10)
    plt.scatter(lk2, lw2n, c='red', s=10)
    plt.plot(lk1, lw1t, label='解析结果', c='orange', ls='--')
    plt.plot(lk2, lw2t, c='orange', ls='--')
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    plt.xlabel(r'$k$')
    plt.ylabel(r'$\omega$', rotation=0)
    plt.legend(bbox_to_anchor=(1, 0.3), loc='upper right', frameon=True)
    plt.title(rf'$N = {N}$；$M_1/M_2 = {t}$；双原子链色散关系曲线')
    plt.ion()
    plt.pause(pause_time)
    plt.close()


time_start = time.time()      # 开始计时

# --------------------- 第一问：一维单原子链色散关系的数值求解 ---------------------
dispertion_num(32)          # 32为谐振子个数，由于自定义的fft适用于任意正整数，谐振子个数也可取任意正整数

# --------------------- 第二问：一维单原子链色散关系的解析求解 ---------------------
dispertion_thm(1000)        # 1000为画图时的取点个数

# ---------------------- 一维单原子链色散关系数值结果&解析结果对比 -------------------------
dispertion_1(64)            # 32为谐振子个数，可取任意正整数

# ---------------------- 一维双原子链色散关系数值结果&解析结果对比 -------------------------
for t in (1.2, 2, 10):      # t = M1/M2
    dispertion_2(32, t)     # 32为双原子链中原子总数，取偶数值


time_end = time.time()          # 结束计时
print(f'运行结束，耗时{time_end - time_start}s')
