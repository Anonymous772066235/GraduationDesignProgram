# File      :all.py
# Author    :WJ
# Function  :
# Time      :2021/02/21
# Version   :
# Amend     :

import LineProcess as lp
import numpy as np
import LaplacianMatrice as lm
from scipy.optimize import linear_sum_assignment
import TransformationMatrix as tf
import Visualization as vs
import LineFit_MaxPoints as lf
import time

start0 = time.time()
# 导入数据
data_dlg = np.loadtxt('..\\data\\Polyline_PCB02_500.txt', delimiter=',')
data_dopp = np.loadtxt('..\\data\\PCB_c0.5_z1_t10.txt', delimiter='\t')
data_dlg0 = data_dlg[:, 0:2]
data_dopp0 = data_dopp[:, 0:2]

# 合并重叠点及邻近点
data_dlg = lf.mergeClosePoints(data_dlg0)
data_dopp = lf.mergeClosePoints(data_dopp0, 1.8)

# 搜索直线
Lines_dlg = lf.searchLines(data_dlg, 16, 6, 0.5)
Lines_dopp = lf.searchLines(data_dopp, 15, 6, 0.5)

# 处理直线
Lines_dlg = lf.processLines(Lines_dlg)
Lines_dlg = lf.mergeLines_ABC(Lines_dlg, X0=np.mean(data_dlg[:, 0]), Y0=np.mean(data_dlg[:, 1]), slope=0.1,
                              intercept=12)
Lines_dopp = lf.processLines(Lines_dopp)
Lines_dopp = lf.mergeLines_ABC(Lines_dopp, X0=np.mean(data_dopp[:, 0]), Y0=np.mean(data_dopp[:, 1]), slope=0.1,
                               intercept=12)

# 可视化直线
lf.showLines(data_dlg, Lines_dlg, np.mean(data_dlg[:, 0]), np.mean(data_dlg[:, 1]),
             name='Lines_dlg' + str(len(Lines_dlg)))
lf.showLines(data_dopp, Lines_dopp, np.mean(data_dopp[:, 0]), np.mean(data_dopp[:, 1]),
             name='Lines_dopp' + str(len(Lines_dopp)))

P_dopp = lp.GetIntersectPointofLines(Lines_dopp)
P_dlg = lp.GetIntersectPointofLines(Lines_dlg)

L_dopp = lm.LaplacianMatrice(P_dopp, sigma=500)
L_dlg = lm.LaplacianMatrice(P_dlg, sigma=500)

U_dopp, Lambda_dopp = lm.LaplacianMatrice_decomposed(L_dopp)
U_dlg, Lambda_dlg = lm.LaplacianMatrice_decomposed(L_dlg)

# 计算相异度矩阵
k = min(len(P_dlg), len(P_dopp))
A = lm.corrlation(U_dopp, U_dlg, k)

# 对相似度矩阵进行二分匹配(删除相异度过大的结果)
row_ind, col_ind = linear_sum_assignment(A)
row, col = lm.DeleteLargeValue(A, row_ind, col_ind, 0.7)

# 根据匹配结果对dopp点云重新排序
P_dopp_new = []
for i in range(len(col)):
    P_dopp_new.append(P_dopp[col, :])
P_dopp_new = np.resize(P_dopp_new, (len(col), 2))

# 可视化直线交点匹配结果
vs.VisualizeMatch(P_dopp, P_dlg, row, col,'直线交点')

# 计算变换矩阵(并对dopp进行变换)
R, T = tf.ca_rt(P_dopp_new, P_dlg, 'MatchingByConPolyPoints_result.txt')
data_dopp = tf.transformation(data_dopp0, R, T, 'dopp_transformed.txt')

# 可视化原始点云配准结果
vs.Visualize2PointClouds(data_dopp, data_dlg0, 'Macth_dlg&dopp', feature1=['blue', 'dopp', '.'],
                         feature2=['red', 'dlg', '.'])

TIME = time.time() - start0
print('\n总耗时:{:.0f} hours {:.0f} minutes {:.0f} seconds'.format(TIME // 3600, TIME % 3600 // 60, TIME % 3600 % 60))
