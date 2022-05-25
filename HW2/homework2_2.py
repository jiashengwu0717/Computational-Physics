# 计算物理第二次大作业第2题源代码
# 运行该程序时，需将'matrix.py'文件和'eigen.py'文件放在该文件的同一目录下
# 运行该程序大约需要200s以上
# 运行过程中图像会自动关闭

pause_time = 5      # pause_time为每张图片的显示时间，默认为5s，可调

import time
import random
from matrix import Matrix, I_Matrix
from eigen import GHR, QR_Wilkinson, Jacobi, power_method_min, dichotomy
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Songti SC']
plt.rcParams['axes.unicode_minus']=False


class generate_H(Matrix):           # 生成哈密顿矩阵H
    def __init__(self, N, h=0.0):   # N为格点个数，h表征外加磁场大小，默认不加外磁场
        matrix = [[0 for c in range(2**N)] for r in range(2**N)]
        for a in range(2 ** N):
            a_2 = bin(a).lstrip('0b').rjust(2 ** N, '0')  # a的二进制表示
            for i in range(N):
                j = (i + 1) % N
                if a_2[-i - 1] == a_2[-j - 1]:  # 相邻格点自旋取向相同
                    matrix[a][a] += 1/4
                else:                           # 相邻格点自旋取向不同
                    matrix[a][a] -= 1/4
                    b = a ^ (2 ** i + 2 ** j)   # 交换第i，j个格点的自旋取向
                    matrix[a][b] += 1/2
            matrix[a][a] += (bin(a).count('1') - N/2) * h   # z方向增加磁场
        super(generate_H, self).__init__(matrix)

class generate_H_mz(Matrix):        # 利用mz守恒进行分块对角化
    def __init__(self, N, n, h=0.0):    # N为格点个数，n为基矢中自旋朝上的态的个数，h表征外加磁场大小
        M, ls = 0, []
        for s in range(2 ** N):
            if bin(s).count('1') == n:
                M += 1              # M表示自旋朝上个数为n的基矢个数
                ls.append(s)        # 列表ls中储存所有自旋朝上个数为n的基矢
        matrix = [[0 for c in range(M)] for r in range(M)]      # 初始化哈密顿矩阵
        for a in range(M):
            for i in range(N):
                j = (i + 1) % N
                a_2 = bin(ls[a]).lstrip('0b').rjust(2 ** N, '0')    # 写出其二进制表示
                if a_2[-i - 1] == a_2[-j - 1]:      # 相邻格点自旋取向相同
                    matrix[a][a] += 1/4
                else:                               # 相邻格点自旋取向不同
                    matrix[a][a] -= 1/4
                    s = ls[a] ^ (2 ** i + 2 ** j)   # 交换第i，j个格点的自旋取向
                    b = ls.index(s)                 # 找出其对应的基矢在列表ls中的位置
                    matrix[a][b] += 1/2
            matrix[a][a] += (n - N / 2) * h     # 外加磁场贡献
        super(generate_H_mz, self).__init__(matrix)

def ground_state(N, h=0.0):     # 计算基态能量和磁化强度
    E, M = 0, []
    for n in range(N + 1):
        T = GHR(generate_H_mz(N, n, h))[0]      # 三对角矩阵
        Emin = dichotomy(T, 1)                  # 二分法求解最小本征值
        if Emin - E < - 10**-5:
            E = Emin
            M = [n - N/2]
        elif abs(Emin - E) < 10**-5:
            M.append(n - N/2)
    return E, M             # E为基态能量，M为磁化强度列表，磁化强度取值可能不止一个

def ground_EMB(N):       # 基态能量随外加磁场的变化曲线；基态磁化曲线
    lh = [0.1 * t for t in range(30)]
    lB, lE, lM = [], [], []
    for h in lh:
        E, M = ground_state(N, h)
        lB += [h for _ in range(len(M))]
        lE += [E for _ in range(len(M))]
        lM += sorted(M, reverse=True)
    plt.plot(lB, lE, label=r'$E \sim B_z$', c='red', ls='--', marker='o', markersize=3)
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    plt.xlabel(r'$B_z$')
    plt.ylabel(r'$E$')
    plt.legend(bbox_to_anchor=(1, 1), frameon=True)
    plt.title(f'N = {N} 基态能量随外加磁场变化曲线')
    plt.ion()
    plt.pause(pause_time)
    plt.close()
    plt.plot(lB, lM, label=r'$M \sim B_z$', c='deepskyblue', ls='--', marker='o', markersize=3)
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    plt.xlabel(r'$B_z$')
    plt.ylabel(r'$M$')
    plt.legend(bbox_to_anchor=(1, 1), frameon=True)
    plt.title(f'N = {N} 基态磁化曲线')
    plt.ion()
    plt.pause(pause_time)
    plt.close()


time_start = time.time()      # 开始计时

# ------------------------------ N=4,直接对角化哈密顿矩阵 -------------------------------
print('------------- 直接对角化H矩阵求解基态能量及其对应的磁化强度 -------------')
N = 4
print(f'N = {N}, h/J = 0')
print('1）经典Jacobi算法')
lE, Q = Jacobi(generate_H(N))
index = lE.index(min(lE))       # 基态能量对应的索引值
v = Q.getCol_list(index + 1)    # 基态归一化本征矢
M = sum(v[k]**2 * (bin(k).count('1') - N/2) for k in range(2**N))   # 磁化强度
print(f'基态能量：{round(lE[index], 5)}')     # 保留5为小数
print(f'基态磁化强度：{round(M, 5)}')          # 保留5为小数
print('')
print('2）带Wilkinson位移的隐式对称QR算法')
lE, Q = QR_Wilkinson(generate_H(N), 0)
index = lE.index(min(lE))       # 基态能量对应的索引值
v = Q.getCol_list(index + 1)    # 基态归一化本征矢
M = sum(v[k]**2 * (bin(k).count('1') - N/2) for k in range(2**N))   # 磁化强度
print(f'基态能量：{round(lE[index], 5)}')     # 保留5为小数
print(f'基态磁化强度：{round(M, 5)}')          # 保留5为小数
print('')

# ------------------------------ N=6,直接对角化哈密顿矩阵 -------------------------------
N = 6
print(f'N = {N}, h/J = 0')
print('带Wilkinson位移的隐式对称QR迭代法计算')
lE, Q = QR_Wilkinson(generate_H(N), 0)
index = lE.index(min(lE))       # 基态能量对应的索引值
v = Q.getCol_list(index + 1)    # 基态归一化本征矢
M = sum(v[k]**2 * (bin(k).count('1') - N/2) for k in range(2**N))   # 磁化强度
print(f'基态能量：{round(lE[index], 5)}')
print(f'基态磁化强度：{round(M, 5)}')
print('')

# ------------------------------------ 不直接对角化 -----------------------------------
print('------------------ 不直接对角化，直接计算基态能量和磁化强度 ----------------')
N = 8
print(f'N = {N}, h/J = 0')
print('1）反幂法')
H = generate_H(N)
p = H.norm(0)           # 行和范数
v = Matrix([[random.uniform(0, 1)] for _ in range(2**N)])        # 初始迭代向量
lam, v = power_method_min(H + I_Matrix(2**N) * (p + 0.1), v, 10**-6)    # 引入0.1为了使迭代更稳定
lam -= p + 0.1
print(f'基态能量：{round(lam, 5)}')
v *= 1/v.norm(2)        # 归一化
M = sum(v[k + 1]**2 * (bin(k).count('1') - N/2) for k in range(2**N))
print(f'基态磁化强度：{round(M, 5)}')
print('')
print('2）分块对角化，并使用Givens-Hessenberg约化和二分法计算各子矩阵最小能量本征值')
E, M = ground_state(N)
print(f'基态能量：{round(E, 5)}')
print('基态磁化强度：' + ', '.join(str(round(x, 5)) for x in M))
print('')

# ------------------------------ N=10时基态能量和磁化强度 --------------------------------
print('----------------- 分块对角化&反幂法——我能计算的最大规模 --------------------')
N = 10
print(f'N={10}, h/J = 0')
E, M = 0, 0
for n in range(N + 1):
    H = generate_H_mz(N, n)
    p = H.norm(0)       # 计算行和范数，确定最小本征值的范围
    v = Matrix([[random.uniform(0, 1)] for _ in range(H.row)])                  # 随机生成初始迭代向量
    lam = power_method_min(H + I_Matrix(H.row) * (p + 0.1), v, 10 ** -6)[0]     # 引入0.1为了使迭代更稳定
    lam -= p + 0.1      # 得到该块状矩阵的最小本征值
    if lam - E < - 10**-10:
        E = lam
        M = n - N/2
print(f'基态能量：{round(E, 5)}')
print(f'基态磁化强度：{round(M, 5)}')
print('')

# -------------------------------- 外加磁场 --------------------------------------
print('------------------ 外加磁场对基态磁化强度的影响 --------------------')
for N in (2, 3, 4, 5, 6):
    print(f'N = {N}')
    lh, lM = [0.1 * t for t in range(30)], []
    for h in lh:
        lM.append(','.join(str(round(x, 3)) for x in ground_state(N, h)[1]))
    print(('+---------')*16 + '+')
    print('|   h/J   |',end = '')
    print(f''.join('{:^9}|'.format(str(round(x, 3))) for x in lh[:15]))
    print(('+---------')*16 + '+')
    print('|    M    |',end = '')
    print(f''.join('{:^9}|'.format(x) for x in lM[:15]))
    print(('+---------')*16 + '+')
    print('|   h/J   |',end = '')
    print(f''.join('{:^9}|'.format(str(round(x, 3))) for x in lh[15:]))
    print(('+---------')*16 + '+')
    print('|    M    |',end = '')
    print(f''.join('{:^9}|'.format(x) for x in lM[15:]))
    print(('+---------')*16 + '+')
    ground_EMB(N)       # 画图
print('')

# -------------------------------- 二维海森堡模型 --------------------------------------
print('---------------------------- N=4时的2维海森堡模型 -----------------------------')
H = generate_H(4) * 2
print('哈密顿矩阵：')
print(H)
lE, Q = QR_Wilkinson(H, 0)      # 使用带Wilkinson位移的对称QR迭代法计算
index = lE.index(min(lE))       # 基态能量对应的索引值
v = Q.getCol_list(index + 1)    # 基态归一化本征矢
M = sum(v[k]**2 * (bin(k).count('1') - 2) for k in range(16))   # 磁化强度
print(f'基态能量：{round(lE[index], 5)}')
print(f'基态磁化强度：{round(M, 5)}')
print('')


time_end = time.time()      # 结束计时
print(f'运行结束，耗时{time_end - time_start}s')
