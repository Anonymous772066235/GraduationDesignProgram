# File      :Match_byCrossPoints.py
# Author    :WJ
# Function  :
# Time      :2021/02/06
# Version   :
# Amend     :
import LineProcess as lp
import numpy as np
import LaplacianMatrice as laplcian
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import Visualization as Vs
import time

start = time.time()

doppLines = np.loadtxt('Lines_dopp15_ok13.txt', delimiter=' ')
dlgLines = np.loadtxt('Lines_dlg16_OK_13.txt', delimiter=' ')

doppPoints = lp.GetIntersectPointofLines(doppLines)
print(len(doppPoints))
dlgPoints = lp.GetIntersectPointofLines(dlgLines)
print(len(dlgPoints))


doppL = laplcian.LaplacianMatrice(doppPoints,sigma=500)
dlgL = laplcian.LaplacianMatrice(dlgPoints,sigma=500)

print('对拉普拉斯矩阵进行谱分解：')

U1, Lambda1 = laplcian.LaplacianMatrice_decomposed(doppL)
U2, Lambda2 = laplcian.LaplacianMatrice_decomposed(dlgL)

print('计算直方图相似度，计算矩阵A：')
k = min(len(doppPoints), len(dlgPoints))

A = laplcian.corrlation(U1, U2, k)

row_ind, col_ind = linear_sum_assignment(A)

print(row_ind)  # 开销矩阵对应的行索引
print(col_ind)  # 对应行索引的最优指派的列索引
print('相异度：')
print(A[row_ind, col_ind])
row, col = laplcian.DeleteLargeValue(A, row_ind, col_ind, 0.7)
print(row)
print(col)
doppPoints_new = []
for i in range(len(col)):
    doppPoints_new.append(doppPoints[col, :])

doppPoints_new = np.resize(doppPoints_new, (len(col), 2))

np.savetxt('CrossPoints_DOPP_OK.txt', doppPoints_new, delimiter=' ')
np.savetxt('CrossPoints_DLG_OK.txt', dlgPoints, delimiter=' ')

print(A[row, col])

plt.figure(figsize=(12, 9))
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
Vs.VisualizePoints(doppPoints, color='red', label='points1')
Vs.VisualizePoints(dlgPoints, color='green', label='points2')
Vs.VisualizeMacth(doppPoints, dlgPoints, row, col)
plt.axis('equal')
plt.xlabel('X')
plt.ylabel('Y')
name = '直线交点匹配'  # 旋转(30)
plt.title(name)
plt.legend(loc='best')
plt.savefig('Match07_' + name + '.png', dpi=300)

TIME = time.time() - start
hours = TIME // 3600
minutes = TIME % 3600 // 60
seconds = TIME % 3600 % 60

print('整个过程耗时------------------------------------------------------')
print('Runningtime:{:.0f} hours {:.0f} minutes {:.0f} seconds'.format(hours, minutes, seconds))
