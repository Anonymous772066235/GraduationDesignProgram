# File      :all.py
# Author    :WJ
# Function  :
# Time      :2021/02/18
# Version   :
# Amend     :

import numpy as np
import ConvexPolygon as cp
import HierarchicalClustering as hc
import ConPolyProcess as cs
import LaplacianMatrice as lm
import time
from scipy.optimize import linear_sum_assignment
import Visualization as vs
import TransformationMatrix as tf


# <editor-fold desc="Method">
def conpoly_points(data, clusters, clusters_num=2):
    P = []
    for k in range(clusters_num):
        ##根据lables中的值是否等于k，重新组成一个True、False的数组
        my_members = clusters == k
        ##X[my_members, 0] 取出my_members对应位置为True的值的横坐标
        data_tem = data[my_members, :]

        indexes = cp.ConvexPolygon(data_tem)
        points = np.array(data_tem[indexes, :], dtype=np.float32)

        while 1:
            max, a0, b0 = cs.maxPoints(points=points)
            if max > 2:
                points = cs.delete_linepoints(points, a0, b0, 3)
            else:
                break
        points = hc.mergeClosePoints(points, 3)
        for i in range(len(points)):
            P.append(points[i, :])

    return np.array(P)


# </editor-fold>

start0 = time.time()
print('求建筑物凸多边形顶点------------------------------------------------------')

# 导入数据
data_dlg = np.loadtxt('..\\data\\Polyline_PCB02_500.txt', delimiter=',')
data_dopp = np.loadtxt('..\\data\\PCB_c1_z5_t20.txt', delimiter='\t')
data_dlg0 = data_dlg[:, 0:2]
data_dopp0 = data_dopp[:, 0:2]

# 设置点云中建筑物聚类数
clusters_num = 2

# 聚类
data_dlg, clusters_dlg = hc.HierarchicalClustering(data_dlg0, clusters_num, 'dlg')
data_dopp, clusters_dopp = hc.HierarchicalClustering(data_dopp0, clusters_num, 'dopp')

# 求每栋建筑物的凸多边形(并对凸多边形顶点进行处理)
P_dlg = conpoly_points(data_dlg, clusters_dlg, clusters_num)
P_dopp = conpoly_points(data_dopp, clusters_dopp, clusters_num)

# 可视化凸多边形顶点
vs.Visualize2PointClouds(data_dlg, P_dlg, 'ConPoly_dlg', feature1=['blue', 'dlg', '.'], feature2=['red', 'vertex', 'o'])
vs.Visualize2PointClouds(data_dopp, P_dopp, 'ConPoly_dlg', feature1=['blue', 'dopp', '.'],
                         feature2=['red', 'vertex', 'o'])

start1 = time.time()
TIME = start1 - start0
print('耗时:{:.0f} hours {:.0f} minutes {:.0f} seconds'.format(TIME // 3600, TIME % 3600 // 60, TIME % 3600 % 60))
print('图匹配------------------------------------------------------')

# 计算拉普拉斯矩阵
B_dlg = lm.LaplacianMatrice(P_dlg)
B_dopp = lm.LaplacianMatrice(P_dopp)

# 对拉普拉斯矩阵进行谱分解
U_dlg, Lambda_dlg = lm.LaplacianMatrice_decomposed(B_dlg)
U_dopp, Lambda_dopp = lm.LaplacianMatrice_decomposed(B_dopp)

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

# 可视化多边形顶点匹配结果
vs.VisualizeMatch(P_dopp, P_dlg, row, col)

# 计算变换矩阵(并对dopp进行变换)
R, T = tf.ca_rt(P_dopp_new, P_dlg, 'MatchingByConPolyPoints_result.txt')
data_dopp = tf.transformation(data_dopp0, R, T, 'dopp_transformed.txt')

# 可视化原始点云配准结果
vs.Visualize2PointClouds(data_dopp, data_dlg0, 'Macth_dlg&dopp', feature1=['blue', 'dopp', '.'],
                         feature2=['red', 'dlg', '.'])

start2 = time.time()
TIME = start2 - start1
print('耗时:{:.0f} hours {:.0f} minutes {:.0f} seconds'.format(TIME // 3600, TIME % 3600 // 60, TIME % 3600 % 60))
print('计算变换矩阵------------------------------------------------------')

TIME = time.time() - start0
print('\n总耗时:{:.0f} hours {:.0f} minutes {:.0f} seconds'.format(TIME // 3600, TIME % 3600 // 60, TIME % 3600 % 60))
