# 计算物理第一次大作业第3题源代码
# 运行该程序约需40s

pause_time = 5      # 每张图片停留时间，默认为5s

# 以下导入的除matplotlib外都是python内置库
import copy
import csv
import time
from math import sqrt, sin, cos, log
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Songti SC']
plt.rcParams['axes.unicode_minus']=False


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
                self.matrix[key - 1] = copy.deepcopy(value)   # 设置矩阵的第i行
            elif self.col == 1:
                self.matrix[key - 1][0] = value     # 设置列向量的第i个分量
        elif isinstance(key, tuple):
            self.matrix[key[0] - 1][key[1] - 1] = value

    def __mul__(self, N):
        if isinstance(N, int) or isinstance(N, float):  # 数乘，且数只能写在右边！
            M = Matrix((self.row, self.col))
            for r in range(self.row):
                for c in range(self.col):
                    M[r, c] = self[r, c] * N
        else:     # 矩阵乘法
            assert N.row == self.col, "维度不匹配，不能相乘"
            M = Matrix((self.row, N.col))
            for r in range(self.row):
                for c in range(N.col):
                    sum = 0
                    for i in range(self.col):
                        sum += self[r, i] * N[i, c]
                    M[r, c] = sum
        return M

    def __str__(self, preference = 16):      # 自定义矩阵打印效果，每个矩阵元默认保留最高精度
        string = '[ '
        if self.row == 1 and self.col == 1:     # 只有一个矩阵元
            string += f'{round(self[1, 1], preference)} ]'
        elif self.row == 1 and self.col > 1:      # 只有一行
            for c in range(1, self.col):
                string += f'{str(round(self[1, c], preference))}  '
            string += f'{str(round(self[1, self.col], preference))} ]'
        elif self.row > 1 and self.col == 1:      # 只有一列
            for r in range(1, self.row):
                string += f'{str(round(self[r, 1], preference))}\n  '
            string += f'{str(round(self[self.row, 1], preference))} ]'
        else:
            for r in range(1, self.row):
                for c in range(1, self.col):
                    string += f'{str(round(self[r, c], preference))}  '
                string += f'{str(round(self[r, self.col], preference))}\n  '
            for c in range(1, self.col):
                string += f'{str(round(self[self.row, c], preference))}  '
            string += f'{str(round(self[self.row, self.col], preference))} ]'
        return string


def solve(A:Matrix, b:Matrix):    # 高斯消元法解方程
    for k in range(1, A.row):
        index = k
        for m in range(k + 1, A.row + 1):   # 找到绝对值最大的元素作为列主元
            if abs(A[m, k]) > abs(A[index, k]):
                index = m   # 该元素的行指标记为index
        if A[index, k] == 0:
            return
        if index != k:
            A[k], A[index] = A[index], A[k]    # 交换两行
            b[k], b[index] = b[index], b[k]
        for i in range(k + 1, A.row + 1):   # 对k行以下每一行进行消元
            A[i, k] /= - A[k, k]     # 倍乘因子
            for j in range(k + 1, A.col + 1):   # 扫描该行每个元素
                A[i, j] += A[i, k] * A[k, j]     # 更新矩阵元素
            b[i] += A[i, k] * b[k]
    for i in range(A.row, 0, -1):   # 回代法解方程
        for j in range(A.col, i, -1):
            b[i] -= A[i, j] * b[j]
        b[i] /= A[i, i]
    return b      # 返回解向量

def lsr(x:list, y:list):      # Least Squares Regression
    k = (len(x) * sum([x[i] * y[i] for i in range(len(x))]) - sum(x)*sum(y)) \
        / (len(x) * sum([t ** 2 for t in x]) - sum(x) ** 2)
    b = (sum(y) - k * sum(x)) / len(x)
    r = (len(x) * sum([x[i] * y[i] for i in range(len(x))]) - sum(x)*sum(y)) \
        / sqrt((len(x)*sum([t**2 for t in x]) - sum(x)**2) * (len(y)*sum([t**2 for t in y]) - sum(y)**2))
    return k, b, r      # 斜率、截距、线性相关系数


time_start = time.time()        # 开始计时


with open('Gyroscope.csv', 'r') as f:
    data = list(csv.reader(f))          # data以嵌套列表的形式存储了Gyroscope.vsc文件的所有数据
lt = [eval(data[i][0]) for i in range(1, len(data))]
lw = [[eval(data[i+1][0]) - eval(data[i][0]), Matrix([[eval(data[i][1])], [eval(data[i][2])], [eval(data[i][3])]])]
      for i in range(2, len(data)-1)]
# 方便第4问使用，同时为解决积分奇异问题，不将data的第一项数据存入lw中

# ----------------------第一问-----------------------
w_yz = [eval(data[i][2]) * eval(data[i][3]) for i in range(1, len(data)-1)]
w_zx = [eval(data[i][3]) * eval(data[i][1]) for i in range(1, len(data)-1)]
w_xy = [eval(data[i][1]) * eval(data[i][2]) for i in range(1, len(data)-1)]
beta_x = [(eval(data[i+1][1]) - eval(data[i][1]))/ (eval(data[i+1][0]) - eval(data[i][0])) for i in range(1, len(data)-1)]
beta_y = [(eval(data[i+1][2]) - eval(data[i][2]))/ (eval(data[i+1][0]) - eval(data[i][0])) for i in range(1, len(data)-1)]
beta_z = [(eval(data[i+1][3]) - eval(data[i][3]))/ (eval(data[i+1][0]) - eval(data[i][0])) for i in range(1, len(data)-1)]

plt.scatter(w_yz, beta_x, label = r'$\beta_x - \omega_y \omega_z$', s = 3)  # \beta_x - \omega_y\omega_z关系图
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')      #设置上边和右边无边框
plt.xlabel(r'$\omega_y \omega_z$')
plt.ylabel(r'$\dot{\omega}_x$')
plt.legend(loc='upper right', frameon=True)
plt.title(r'$\beta_x - \omega_y \omega_z$关系图')
plt.ion()
plt.pause(pause_time)
plt.close()

plt.scatter(w_zx, beta_y, label = r'$\beta_y - \omega_z \omega_x$', s = 3)     # \beta_y - \omega_z\omega_x关系图
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')      #设置上边和右边无边框
plt.xlabel(r'$\omega_z \omega_x$')
plt.ylabel(r'$\dot{\omega}_y$')
plt.legend(loc='upper left', frameon=True)
plt.title(r'$\beta_y - \omega_z \omega_x$关系图')
plt.ion()
plt.pause(pause_time)
plt.close()

plt.scatter(w_xy, beta_z, label = r'$\beta_z - \omega_x \omega_y$', s = 3)     # \beta_z - \omega_x\omega_y关系图
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')      #设置上边和右边无边框
plt.xlabel(r'$\omega_x \omega_y$')
plt.ylabel(r'$\dot{\omega}_z$')
plt.legend(loc='upper left', frameon=True)
plt.title(r'$\beta_z - \omega_x \omega_y$关系图')
plt.ion()
plt.pause(pause_time)
plt.close()

llx = [log(abs(beta_x[i] / w_yz[i])+1) for i in range(len(beta_x))]
lly = [log(abs(beta_y[i] / w_zx[i])+1) for i in range(len(beta_y))]
llz = [log(abs(beta_z[i] / w_xy[i])+1) for i in range(len(beta_z))]
plt.plot(lt[:-1], llx, label = r'$ln(|\beta_x / \omega_y \omega_z|+1) - t$', c = 'red')
plt.plot(lt[:-1], lly, label = r'$ln(|\beta_y / \omega_z \omega_x|+1) - t$', c = 'orange')
plt.plot(lt[:-1], llz, label = r'$ln(|\beta_z / \omega_x \omega_y|+1) - t$', c = 'blue')
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')      #设置上边和右边无边框
plt.xlabel(r'$t$')
plt.ylabel(r'$ln(|\dfrac{\beta_{\alpha}}{\omega_{\beta} \omega_{\gamma}}|+1)$',labelpad=20,rotation = 0)
plt.legend(bbox_to_anchor=(0.83, 1), loc='upper right', frameon=True)
plt.title(r'$ln(|\dfrac{\beta_{\alpha}}{\omega_{\beta} \omega_{\gamma}}|+1) - t$ 关系图')
plt.annotate(r'$t = 1.370$', xy=(1.37, 0.5), xytext=(1.1, -1.2), fontsize=10,
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.3'))
plt.annotate(r'$t = 1.750$', xy=(1.75, 0.4), xytext=(1.5, -1.6), fontsize=10,
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3'))
plt.ion()
plt.pause(pause_time)
plt.close()


# ----------------------第二问-----------------------
print('----------------- 拟合得到的斜率k-----------------')
w_yz = w_yz[570 : 730]
w_zx = w_zx[570 : 730]
w_xy = w_xy[570 : 730]
beta_x = beta_x[570 : 730]
beta_y = beta_y[570 : 730]
beta_z = beta_z[570 : 730]
k_x, b_x ,r_x = lsr(w_yz, beta_x)
k_y, b_y, r_y = lsr(w_zx, beta_y)
k_z, b_z, r_z = lsr(w_xy, beta_z)
print(f'线性拟合得到：k_x = {k_x}， 线性相关系数为：{r_x}')
print(f'线性拟合得到：k_y = {k_y}， 线性相关系数为：{r_y}')
print(f'线性拟合得到：k_z = {k_z}， 线性相关系数为：{r_z}')
print('')


# ----------------------第三问-----------------------
print('---------------手机各轴转动惯量之比---------------')
A = Matrix([[k_x ** 2 + 2, -k_x + k_y - 1], [-k_x + k_y - 1, k_y ** 2 + 2]])
b = Matrix([[-k_x + k_z + 1], [k_y - k_z + 1]])
I = solve(A, b)
print(f'I_x : I_y : I_z = {I[1]} : {I[2]} : 1')
print('')


# ----------------------第四问-----------------------
theta0 = eval(data[1][1]) * eval(data[2][0])    # 初始值，取为omega_x t
l_theta = [theta0]                  # 记录theta随时间变化关系的的列表
l_varphi = [0]                      # 记录varphi随时间变化关系的的列表
l_psi = [0]                         # 记录psi随时间变化关系的的列表
for x in lw:
    A = Matrix([[cos(l_psi[-1]), sin(l_theta[-1]) * sin(l_psi[-1]), 0],
                [-sin(l_psi[-1]), sin(l_theta[-1]) * cos(l_psi[-1]), 0],
                [0, cos(l_theta[-1]), 1]])
    Euler = solve(A, x[1])                  # 解得的是三个欧拉角对时间的导数
    l_theta.append(l_theta[-1] + Euler[1] * x[0])
    l_varphi.append(l_varphi[-1] + Euler[2] * x[0])
    l_psi.append(l_psi[-1] + Euler[3] * x[0])
l_theta = [x - theta0 for x in l_theta]     # 减去最初设定的初始值，不减问题也不大

plt.scatter(lt[1:], l_theta, label = r'$\theta - t$', s = 5)
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')      #设置上边和右边无边框
plt.xlabel(r'$t$')
plt.ylabel(r'$\theta$')
plt.legend(loc='upper right', frameon=True)
plt.title(r'$\theta - t$变化关系图')
plt.ion()
plt.pause(pause_time)
plt.close()

plt.scatter(lt[1:], l_varphi, label = r'$\varphi - t$', s = 5)
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')      #设置上边和右边无边框
plt.xlabel(r'$t$')
plt.ylabel(r'$\varphi$')
plt.legend(loc='upper right', frameon=True)
plt.title(r'$\varphi - t$变化关系图')
plt.ion()
plt.pause(pause_time)
plt.close()

plt.scatter(lt[1:], l_psi, label = r'$\psi - t$', s = 5)
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')      #设置上边和右边无边框
plt.xlabel(r'$t$')
plt.ylabel(r'$\psi$')
plt.legend(loc='upper left', frameon=True)
plt.title(r'$\psi - t$变化关系图')
plt.ion()
plt.pause(pause_time)
plt.close()


time_end = time.time()      # 结束计时
print(f'运行结束，耗时{time_end - time_start}s')

