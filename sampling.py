from math import sqrt, log, e
from random import uniform
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Songti SC']
plt.rcParams['axes.unicode_minus']=False


def Marsaglia():        # 生成高斯分布的随机数
    while True:
        u, v = uniform(0, 1), uniform(0, 1)
        w = (2*u-1)**2 + (2*v-1)**2
        if w <= 1:
            break
    z = sqrt(-2 * log(w) / w)
    return (2*u-1) * z, (2*v-1) * z

def put(l, x):
    if x < -5:
        l[0] += 1
    elif x > 5:
        l[-1] += 1
    else:
        l[int((10 * x) // 1) + 51] += 1

ln = np.linspace(-5, 5, 102)
lcount = [0 for i in range(102)]
for _ in range(500000):
    x, y = Marsaglia()
    put(lcount, x)
    put(lcount, y)

plt.plot(ln, lcount, c = 'red')
plt.show()





