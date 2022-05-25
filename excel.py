import xlrd
import openpyxl
import pandas as pd
import matplotlib.pyplot as plt

'''
df = pd.read_excel('wavefunctionN6d3.xlsx', engine='openpyxl')
data = df.values

l0 = [data[i][0] for i in range(2, 203)]
for j in [6, 12]:
    l = [data[i][j] for i in range(2, 203)]
    plt.plot(l0, l)
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')      #设置上边和右边无边框
    ax.xaxis.set_ticks_position('bottom')   #设置x坐标刻度数字或名称的位置
    ax.spines['bottom'].set_position(('data', 0))   #设置边框位置
    plt.title(f'N=6,d/a=1/3,wave-{j}')
    plt.show()
'''


lx1  = [13, 14, 15]
ly1 = [1.4, 1.4, 1.6]
lu1 = [13]
lv1 = [1.0]
lx2 = [14, 15, 16, 17, 18, 19, 20]
ly2 = [2.4, 2.3, 2.0, 2.2, 2.2, 2.1, 2.2]
lu2 = [14, 15, 16, 17, 18]
lv2 = [1.3, 1.4, 1.6, 1.6, 1.8]
lx3 = [18, 19, 20, 21, 22, 23]
ly3 = [3.1, 2.9, 2.4, 2.6, 2.6, 2.9]
lu3 = [18, 19, 20, 21, 22]
lv3 = [2.0, 2.0, 2.1, 2.1, 2.2]
lx4 = [21, 23, 24]
ly4 = [3.9, 3.3, 3.3]
lu4 = [21, 24]
lv4 = [3.5, 2.5]
plt.scatter(lx1, ly1, c = 'r', marker = '^', label = 'N = 1, up')
plt.scatter(lu1, lv1, c = 'r', marker = 'v', label = 'N = 1, down')
plt.scatter(lx2, ly2, c = 'y', marker = '^', label = 'N = 2, up')
plt.scatter(lu2, lv2, c = 'y', marker = 'v', label = 'N = 2, down')
plt.scatter(lx3, ly3, c = 'blue', marker = '^', label = 'N = 3, up')
plt.scatter(lu3, lv3, c = 'blue', marker = 'v', label = 'N = 3, down')
plt.scatter(lx4, ly4, c = 'black', marker = '^', label = 'N = 4, up')
plt.scatter(lu4, lv4, c = 'black', marker = 'v', label = 'N = 4, down')
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')      #设置上边和右边无边框
plt.xlabel('f/Hz')
plt.ylabel('Upp/V')
plt.legend(loc='upper left', frameon=True)
plt.title(f'Upp-f')
plt.show()






