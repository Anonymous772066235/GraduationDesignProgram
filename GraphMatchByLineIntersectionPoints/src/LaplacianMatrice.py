# File      :LaplacianMatrice.py
# Author    :WJ
# Function  :
# Time      :2021/01/07
# Version   :
# Amend     :
import numpy as np
import math
from sympy import Matrix, GramSchmidt
import matplotlib.pyplot as plt
import Visualization as Vs
import time
import CompareHist as ch
from scipy.optimize import linear_sum_assignment
import Interference as it


def LaplacianMatrice(Pset, sigma=200):
    # 计算点云的拉普拉斯矩阵
    # Pset为二维点云，sigma为距离参数
    L = np.zeros((len(Pset), len(Pset)))
    for i in range(0, len(Pset)):
        for j in range(0, len(Pset)):
            if i != j:
                L[i, j] = -math.exp(np.linalg.norm(Pset[i, :] - Pset[j, :]) / sigma)
    for i in range(0, len(Pset)):
        L[i, i] = -np.sum(L[:, i])
    return L


def LaplacianMatrice_decomposed(LM):
    # 对点云的拉普拉斯矩阵进行谱分解
    eigenvalue, V = np.linalg.eig(LM)  # 求特征值与特征向量
    Lambda = np.diag(eigenvalue)  # 特征值对角矩阵
    O = []
    for i in range(0, len(V[:, 0])):
        O.append(Matrix(V[:, i]))
    U_T = GramSchmidt(O, True)  # 特征向量施密特正交化
    U = np.transpose(U_T)  # 此时U是3D数组，输出时需要降一维度为2D数组
    return U[0, :, :], Lambda


def HistCorrelation(U1, U2, k):
    U1k = U1
    U2k = U2
    # 相关度计算
    A = np.zeros((k, k))
    for i in range(0, k):
        for j in range(0, k):
            A[j, i] = ch.calaHistCorrelation(U1k[i, :], U2k[j, :])
    return A


def corrlation(U1, U2, k):
    U1k = U1[:, 0:k]
    U2k = U2[:, 0:k]
    # 相关度计算
    A = np.zeros((k, k))
    for i in range(0, k):
        for j in range(0, k):
            A[j, i] = ch.calcSimilarity(U1k[i, :], U2k[j, :])
            # A[j,i]=ch.calcSimilarity(U1k[:,i], U2k[:,j])
    return A


def DeleteLargeValue(A, row, col, maxValue):
    row = np.array(row)
    col = np.array(col)
    a = A[row, col]
    d = []
    for i in range(len(a)):
        if a[i] > maxValue:
            d.append(i)
    d.reverse()
    for j in range(len(d)):
        row = np.delete(row, d[j])
        col = np.delete(col, d[j])
    return row, col


if __name__ == '__main__':
    start = time.time()
    set0 = np.loadtxt('crossPoints_dopp.txt', delimiter=' ')
    set1 = np.loadtxt('crossPoints_dlg.txt', delimiter=' ')
    set0 = set0[:, 0:2]
    set1 = set1[:, 0:2]





    # set0 = np.loadtxt('truedata_1.txt', delimiter=',')
    # set0 = set0[:, 0:2]
    # set1 = set0[:, 0:2]
    # set0 = np.loadtxt('truedata_1.txt', delimiter=',')
    # set1 = np.loadtxt('PCB_c1_z5_t20.txt', delimiter='\t')
    # set0 = set0[:, 0:2]
    # set1 = set1[:, 0:2]

    # b1 = cp.ConvexPolygon(set0)
    # b2 = cp.ConvexPolygon(set1)
    # set0 = np.array(set0[b1, :], dtype=np.float32)
    # set1 = np.array(set1[b2, :], dtype=np.float32)
    # set0 = cp.minTrinagle(set0)
    # set1 = cp.minTrinagle(set1)
    # set1 = it.Translation(set1, 200, 5)
    # set1=it.Deletion(set1,5)
    # set1 = it.Shuffle(set1)

    print('求拉普拉斯矩阵：')
    B1 = LaplacianMatrice(set0)

    B2 = LaplacianMatrice(set1)

    TIME = time.time() - start
    hours = TIME // 3600
    minutes = TIME % 3600 // 60
    seconds = TIME % 3600 % 60


    print('求拉普拉斯矩阵耗时------------------------------------------------------')
    print('Runningtime:{:.0f} hours {:.0f} minutes {:.0f} seconds'.format(hours, minutes, seconds))

    start1 = time.time()
    print('对拉普拉斯矩阵进行谱分解：')

    U1, Lambda1 = LaplacianMatrice_decomposed(B1)
    U2, Lambda2 = LaplacianMatrice_decomposed(B2)

    np.savetxt('U1.csv', U1, delimiter=',')
    np.savetxt('U2.csv', U2, delimiter=',')

    TIME = time.time() - start1
    hours = TIME // 3600
    minutes = TIME % 3600 // 60
    seconds = TIME % 3600 % 60

    print('谱分解耗时------------------------------------------------------')
    print('Runningtime:{:.0f} hours {:.0f} minutes {:.0f} seconds'.format(hours, minutes, seconds))

    start2 = time.time()
    print('计算直方图相似度，计算矩阵A：')
    k = min(len(set0), len(set1))
    # A = HistCorrelation(U1, U2, k)
    A=corrlation(U1,U2,k)
    TIME = time.time() - start2
    hours = TIME // 3600
    minutes = TIME % 3600 // 60
    seconds = TIME % 3600 % 60

    print('求矩阵A耗时------------------------------------------------------')
    print('Runningtime:{:.0f} hours {:.0f} minutes {:.0f} seconds'.format(hours, minutes, seconds))


    print(A)
    np.savetxt('A.csv', A, delimiter=',')

    row_ind, col_ind = linear_sum_assignment(A)

    print(row_ind)  # 开销矩阵对应的行索引
    print(col_ind)  # 对应行索引的最优指派的列索引
    print('相异度：')
    print(A[row_ind, col_ind])
    row, col = DeleteLargeValue(A, row_ind, col_ind, 0.7)
    print(row)
    print(col)
    dopp_new=[]
    for i in range(len(col)):
        dopp_new.append(set0[col,:])
    dopp_new=np.resize(dopp_new,(len(col),2))

    print(dopp_new)
    np.savetxt('crossPoints_dopp02.txt',dopp_new,delimiter=' ')

    print(A[row, col])

    plt.figure(figsize=(12, 9))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    Vs.VisualizePoints(set0, color='red', label='points1')
    Vs.VisualizePoints(set1, color='green', label='points2')
    Vs.VisualizeMacth(set0, set1, row, col)
    plt.axis('equal')
    plt.xlabel('X')
    plt.ylabel('Y')
    name = '直线交点'  # 旋转(30)
    plt.title(name)
    plt.legend(loc='best')
    plt.savefig('Match06_' + name + '.png', dpi=300)

    # # <editor-fold desc="检测重新排序是否正确">
    # plt.figure(figsize=(12, 9))
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False
    # Vs.VisualizePoints(dopp_new, color='red', label='points1')
    # Vs.VisualizePoints(set1, color='green', label='points2')
    # Vs.VisualizeMacth(dopp_new, set1, row, row)
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # name = '直线交点'  # 旋转(30)
    # plt.title(name)
    # plt.legend(loc='best')
    # plt.savefig('Match06_' + name + '.png', dpi=300)
    # # </editor-fold>


    TIME = time.time() - start
    hours = TIME // 3600
    minutes = TIME % 3600 // 60
    seconds = TIME % 3600 % 60

    print('整个过程耗时------------------------------------------------------')
    print('Runningtime:{:.0f} hours {:.0f} minutes {:.0f} seconds'.format(hours, minutes, seconds))
