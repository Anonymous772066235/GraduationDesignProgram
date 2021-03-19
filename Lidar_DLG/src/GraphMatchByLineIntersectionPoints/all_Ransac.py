# File      :all_02.py
# Author    :WJ
# Function  :
# Time      :2021/02/21
# Version   :
# Amend     :
import time
import numpy as np
from scipy.optimize import linear_sum_assignment

try:
    import GraphMatchByLineIntersectionPoints.LineProcess as lp
    import GraphMatchByLineIntersectionPoints.LaplacianMatrice as lm
    import GraphMatchByLineIntersectionPoints.TransformationMatrix as tf
    import GraphMatchByLineIntersectionPoints.Visualization as vs
    import GraphMatchByLineIntersectionPoints.LineFit_Ransac as lf
except:
    import LineProcess as lp
    import LaplacianMatrice as lm
    import TransformationMatrix as tf
    import Visualization as vs
    import LineFit_Ransac as lf


def run(data_dlg, data_dopp, name='ABC'):
    if name == 'PCB':
        # 导入数据
        data_dlg0 = data_dlg[:, 0:2]
        data_dopp0 = data_dopp[:, 0:2]

        # 合并重叠点及邻近点
        data_dlg = lf.mergeClosePoints(data_dlg0)
        data_dopp = lf.mergeClosePoints(data_dopp0, 1.8)

        # 搜索直线
        Lines_dlg = lf.searchLines(data_dlg, 17, 2000, 6, 0.5)
        Lines_dopp = lf.searchLines(data_dopp, 15, 2000, 6, 0.5)

        # 处理直线
        Lines_dlg = lf.processLines(Lines_dlg)
        Lines_dlg = lf.mergeLines_ABC(Lines_dlg, X0=np.mean(data_dlg[:, 0]), Y0=np.mean(data_dlg[:, 1]), slope=0.1,
                                      intercept=17)
        Lines_dopp = lf.processLines(Lines_dopp)
        Lines_dopp = lf.mergeLines_ABC(Lines_dopp, X0=np.mean(data_dopp[:, 0]), Y0=np.mean(data_dopp[:, 1]), slope=0.1,
                                       intercept=12)

        # 可视化直线
        lf.showLines(data_dlg, Lines_dlg, np.mean(data_dlg[:, 0]), np.mean(data_dlg[:, 1]),
                     name='Lines_dlg_ransac' + str(len(Lines_dlg)))
        lf.showLines(data_dopp, Lines_dopp, np.mean(data_dopp[:, 0]), np.mean(data_dopp[:, 1]),
                     name='Lines_dopp_ransac' + str(len(Lines_dopp)))

        P_dopp = lp.GetIntersectPointofLines(Lines_dopp)
        P_dlg = lp.GetIntersectPointofLines(Lines_dlg)

        L_dopp = lm.LaplacianMatrice(P_dopp, sigma=200)
        L_dlg = lm.LaplacianMatrice(P_dlg, sigma=200)

        U_dopp, Lambda_dopp = lm.LaplacianMatrice_decomposed(L_dopp)
        U_dlg, Lambda_dlg = lm.LaplacianMatrice_decomposed(L_dlg)

        # 计算相异度矩阵
        k = min(len(P_dlg), len(P_dopp))
        A = lm.corrlation(U_dopp, U_dlg, k)

        # 对相似度矩阵进行二分匹配(删除相异度过大的结果)
        row_ind, col_ind = linear_sum_assignment(A)
        row, col = lm.DeleteLargeValue(A, row_ind, col_ind, 0.8)

        # 根据匹配结果对点云重新排序
        P_dlg_new = lm.resort_clouds(P_dlg, row)
        P_dopp_new = lm.resort_clouds(P_dopp, col)

        # 可视化直线交点匹配结果
        vs.VisualizeMatch(P_dopp, P_dlg, row, col, '直线交点_ransac')

        # 计算变换矩阵(并对dopp进行变换)
        R, T = tf.ca_rt(P_dopp_new, P_dlg_new)
        data_dopp = tf.transformation(data_dopp0, R, T)

        # 可视化原始点云配准结果
        vs.Visualize2PointClouds(data_dopp, data_dlg0, 'Macth_dlg&dopp_ransac', feature1=['blue', 'dopp', '.'],
                                 feature2=['red', 'dlg', '.'])
        return R, T, data_dopp

    elif name == 'ABC':

        # 导入数据
        data_dlg0 = data_dlg[:, 0:2]
        data_dopp0 = data_dopp[:, 0:2]

        # 合并重叠点及邻近点
        data_dlg = lf.mergeClosePoints(data_dlg0)
        data_dopp = lf.mergeClosePoints(data_dopp0)

        # 搜索直线
        Lines_dlg = lf.searchLines(data_dlg, 7, 3000, 4, 0.1)
        Lines_dopp = lf.searchLines(data_dopp, 7, 3000, 4, 0.1)
        # Lines_dlg = lf.searchLines(data_dlg, 12, 4000, 4, 0.2)
        # Lines_dopp = lf.searchLines(data_dopp,12, 4000, 4,0.2)

        # 处理直线
        Lines_dlg = lf.processLines(Lines_dlg)
        Lines_dlg = lf.mergeLines_ABC(Lines_dlg, X0=np.mean(data_dlg[:, 0]), Y0=np.mean(data_dlg[:, 1]), slope=0.1,
                                      intercept=15)
        Lines_dopp = lf.processLines(Lines_dopp)
        Lines_dopp = lf.mergeLines_ABC(Lines_dopp, X0=np.mean(data_dopp[:, 0]), Y0=np.mean(data_dopp[:, 1]), slope=0.1,
                                       intercept=15)

        # 可视化直线
        lf.showLines(data_dlg, Lines_dlg, np.mean(data_dlg[:, 0]), np.mean(data_dlg[:, 1]),
                     name='Lines_dlg_ransac_abc' + str(len(Lines_dlg)))
        lf.showLines(data_dopp, Lines_dopp, np.mean(data_dopp[:, 0]), np.mean(data_dopp[:, 1]),
                     name='Lines_dopp_ransac_abc' + str(len(Lines_dopp)))

        P_dopp = lp.GetIntersectPointofLines(Lines_dopp)
        P_dlg = lp.GetIntersectPointofLines(Lines_dlg)

        L_dopp = lm.LaplacianMatrice(P_dopp, sigma=200)
        L_dlg = lm.LaplacianMatrice(P_dlg, sigma=200)

        U_dopp, Lambda_dopp = lm.LaplacianMatrice_decomposed(L_dopp)
        U_dlg, Lambda_dlg = lm.LaplacianMatrice_decomposed(L_dlg)

        # 计算相异度矩阵
        k = min(len(P_dlg), len(P_dopp))
        A = lm.corrlation(U_dopp, U_dlg, k)

        # 对相似度矩阵进行二分匹配(删除相异度过大的结果)
        row_ind, col_ind = linear_sum_assignment(A)
        row, col = lm.DeleteLargeValue(A, row_ind, col_ind, 0.9)

        # 根据匹配结果对点云重新排序
        P_dlg_new = lm.resort_clouds(P_dlg, row)
        P_dopp_new = lm.resort_clouds(P_dopp, col)

        # 可视化直线交点匹配结果
        vs.VisualizeMatch(P_dopp, P_dlg, row, col, '直线交点_ransac_abc')

        # 计算变换矩阵(并对dopp进行变换)
        R, T = tf.ca_rt(P_dopp_new, P_dlg_new)
        data_dopp = tf.transformation(data_dopp0, R, T)

        # 可视化原始点云配准结果
        vs.Visualize2PointClouds(data_dopp, data_dlg0, 'Macth_dlg&dopp_ransac_abc', feature1=['blue', 'dopp', '.'],
                                 feature2=['red', 'dlg', '.'])
        return R, T, data_dopp


if __name__ == "__main__":

    # name='PCB'
    name = 'ABC'

    if name == 'PCB':
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
        Lines_dlg = lf.searchLines(data_dlg, 17, 2000, 6, 0.5)
        Lines_dopp = lf.searchLines(data_dopp, 15, 2000, 6, 0.5)

        # 处理直线
        Lines_dlg = lf.processLines(Lines_dlg)
        Lines_dlg = lf.mergeLines_ABC(Lines_dlg, X0=np.mean(data_dlg[:, 0]), Y0=np.mean(data_dlg[:, 1]), slope=0.1,
                                      intercept=12)
        Lines_dopp = lf.processLines(Lines_dopp)
        Lines_dopp = lf.mergeLines_ABC(Lines_dopp, X0=np.mean(data_dopp[:, 0]), Y0=np.mean(data_dopp[:, 1]), slope=0.1,
                                       intercept=12)

        # 可视化直线
        lf.showLines(data_dlg, Lines_dlg, np.mean(data_dlg[:, 0]), np.mean(data_dlg[:, 1]),
                     name='Lines_dlg_ransac' + str(len(Lines_dlg)))
        lf.showLines(data_dopp, Lines_dopp, np.mean(data_dopp[:, 0]), np.mean(data_dopp[:, 1]),
                     name='Lines_dopp_ransac' + str(len(Lines_dopp)))

        P_dopp = lp.GetIntersectPointofLines(Lines_dopp)
        P_dlg = lp.GetIntersectPointofLines(Lines_dlg)

        L_dopp = lm.LaplacianMatrice(P_dopp, sigma=200)
        L_dlg = lm.LaplacianMatrice(P_dlg, sigma=200)

        U_dopp, Lambda_dopp = lm.LaplacianMatrice_decomposed(L_dopp)
        U_dlg, Lambda_dlg = lm.LaplacianMatrice_decomposed(L_dlg)

        # 计算相异度矩阵
        k = min(len(P_dlg), len(P_dopp))
        A = lm.corrlation(U_dopp, U_dlg, k)

        # 对相似度矩阵进行二分匹配(删除相异度过大的结果)
        row_ind, col_ind = linear_sum_assignment(A)
        row, col = lm.DeleteLargeValue(A, row_ind, col_ind, 0.9)

        # 根据匹配结果对点云重新排序
        P_dlg_new = lm.resort_clouds(P_dlg, row)
        P_dopp_new = lm.resort_clouds(P_dopp, col)

        # 可视化直线交点匹配结果
        vs.VisualizeMatch(P_dopp, P_dlg, row, col, '直线交点_ransac')

        # 计算变换矩阵(并对dopp进行变换)
        R, T = tf.ca_rt(P_dopp_new, P_dlg_new, 'MatchingByLineIntersectionPoints_result_ransac.txt')
        data_dopp = tf.transformation(data_dopp0, R, T, 'dopp_transformed_ransac.txt')

        # 可视化原始点云配准结果
        vs.Visualize2PointClouds(data_dopp, data_dlg0, 'Macth_dlg&dopp_ransac', feature1=['blue', 'dopp', '.'],
                                 feature2=['red', 'dlg', '.'])

        TIME = time.time() - start0
        print('\n总耗时:{:.0f} hours {:.0f} minutes {:.0f} seconds'.format(TIME // 3600, TIME % 3600 // 60,
                                                                        TIME % 3600 % 60))

    elif name == 'ABC':
        start0 = time.time()
        # 导入数据
        data_dlg = np.loadtxt('..\\data\\Polyline_ABC.txt', delimiter=',')
        data_dopp = np.loadtxt('..\\data\\ABC_c1_z10_t10.txt', delimiter='\t')
        data_dlg0 = data_dlg[:, 0:2]
        data_dopp0 = data_dopp[:, 0:2]

        # 合并重叠点及邻近点
        data_dlg = lf.mergeClosePoints(data_dlg0)
        data_dopp = lf.mergeClosePoints(data_dopp0)

        # 搜索直线
        Lines_dlg = lf.searchLines(data_dlg, 7, 3000, 4, 0.1)
        Lines_dopp = lf.searchLines(data_dopp, 7, 3000, 4, 0.1)
        # Lines_dlg = lf.searchLines(data_dlg, 12, 4000, 4, 0.2)
        # Lines_dopp = lf.searchLines(data_dopp,12, 4000, 4,0.2)

        # 处理直线
        Lines_dlg = lf.processLines(Lines_dlg)
        Lines_dlg = lf.mergeLines_ABC(Lines_dlg, X0=np.mean(data_dlg[:, 0]), Y0=np.mean(data_dlg[:, 1]), slope=0.1,
                                      intercept=15)
        Lines_dopp = lf.processLines(Lines_dopp)
        Lines_dopp = lf.mergeLines_ABC(Lines_dopp, X0=np.mean(data_dopp[:, 0]), Y0=np.mean(data_dopp[:, 1]), slope=0.1,
                                       intercept=15)

        # 可视化直线
        lf.showLines(data_dlg, Lines_dlg, np.mean(data_dlg[:, 0]), np.mean(data_dlg[:, 1]),
                     name='Lines_dlg_ransac_abc' + str(len(Lines_dlg)))
        lf.showLines(data_dopp, Lines_dopp, np.mean(data_dopp[:, 0]), np.mean(data_dopp[:, 1]),
                     name='Lines_dopp_ransac_abc' + str(len(Lines_dopp)))

        P_dopp = lp.GetIntersectPointofLines(Lines_dopp)
        P_dlg = lp.GetIntersectPointofLines(Lines_dlg)

        L_dopp = lm.LaplacianMatrice(P_dopp, sigma=200)
        L_dlg = lm.LaplacianMatrice(P_dlg, sigma=200)

        U_dopp, Lambda_dopp = lm.LaplacianMatrice_decomposed(L_dopp)
        U_dlg, Lambda_dlg = lm.LaplacianMatrice_decomposed(L_dlg)

        # 计算相异度矩阵
        k = min(len(P_dlg), len(P_dopp))
        A = lm.corrlation(U_dopp, U_dlg, k)

        # 对相似度矩阵进行二分匹配(删除相异度过大的结果)
        row_ind, col_ind = linear_sum_assignment(A)
        row, col = lm.DeleteLargeValue(A, row_ind, col_ind, 0.9)

        # 根据匹配结果对点云重新排序
        P_dlg_new = lm.resort_clouds(P_dlg, row)
        P_dopp_new = lm.resort_clouds(P_dopp, col)

        # 可视化直线交点匹配结果
        vs.VisualizeMatch(P_dopp, P_dlg, row, col, '直线交点_ransac_abc')

        # 计算变换矩阵(并对dopp进行变换)
        R, T = tf.ca_rt(P_dopp_new, P_dlg_new, 'MatchingByLineIntersectionPoints_result_ransac_abc.txt')
        data_dopp = tf.transformation(data_dopp0, R, T, 'dopp_transformed_ransac_abc.txt')

        # 可视化原始点云配准结果
        vs.Visualize2PointClouds(data_dopp, data_dlg0, 'Macth_dlg&dopp_ransac_abc', feature1=['blue', 'dopp', '.'],
                                 feature2=['red', 'dlg', '.'])

        TIME = time.time() - start0
        print('\n总耗时:{:.0f} hours {:.0f} minutes {:.0f} seconds'.format(TIME // 3600, TIME % 3600 // 60,
                                                                        TIME % 3600 % 60))
