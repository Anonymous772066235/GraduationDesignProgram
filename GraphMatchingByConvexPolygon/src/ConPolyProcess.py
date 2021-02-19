# File      :ConPolyProcess.py
# Author    :WJ
# Function  :
# Time      :2021/01/22
# Version   :
# Amend     :

import matplotlib.pyplot as plt
import numpy as np
import math
import time


def maxPoints(points):
    # write your code here
    # 神坑，还有一个点的情况，哎
    if len(points) == 1:
        return 1
    # 任意两个点练成一条直线，判断是否再这条直线上
    # 获取两个点
    max = 0
    a = 0
    a_max = 0
    b_max = 0
    MAX = []
    A = []
    B = []
    for i in range(len(points)):
        for j in range(len(points)):
            if i != j:
                pointcount = 0
                # 拿到两个点，获得矢量
                # 排除一种x2-x1 = 0 的情况
                b = points[j, 0] - points[i, 0]
                if b == 0:
                    # 遍历找到都在这条直线上的点
                    for t in range(len(points)):
                        if points[t, 0] == points[i, 0]:
                            pointcount = pointcount + 1
                else:
                    # 根据公式 a= (y2-y1)/(x2-x1)
                    a = (points[j, 1] - points[i, 1]) / b
                    b = -a * points[i, 0] + points[i, 1]
                    # 遍历其他点是否再这条直线上
                    pointcount = 1
                    for k in range(len(points)):
                        if (points[k, 0] != points[i, 0]):
                            if abs(points[k, 1] - (a * (points[k, 0] - points[i, 0]) + points[i, 1])) / math.sqrt(
                                    1 + a * a) < 1:
                                pointcount = pointcount + 1

                if max < pointcount:
                    max = pointcount
                    a_max = a
                    b_max = b
        MAX.append(max)
        A.append(a_max)
        B.append(b_max)
    return MAX[np.argmax(MAX)], A[np.argmax(MAX)], B[np.argmax(MAX)]

def delete_linepoints(data1, a, b, distance=1):
    K = []
    for k in range(len(data1)):
        if abs(data1[k, 1] - (a * data1[k, 0] + b)) / math.sqrt(1 + a * a) < distance:
            K.append(k)
    K.sort(reverse=True)
    line = data1[K, :]
    x01 = np.argmin(line[:, 0])
    x02 = np.argmax(line[:, 0])
    if x01 > x02:
        K = np.delete(K, x01, axis=0)  # 从线段点云中删除线段起始点
        K = np.delete(K, x02, axis=0)
    else:
        K = np.delete(K, x02, axis=0)  # 从线段点云中删除线段起始点
        K = np.delete(K, x01, axis=0)

    for i in range(len(K)):
        data1 = np.delete(data1, K[i], axis=0)
    return data1

def delete_closepoints(data, distance):
    for i in range(len(data)):
        de = []
        for j in range(i + 1, len(data)):
            if np.linalg.norm(data[i, :] - data[j, :]) < distance:
                de.append(j)
        if len(de) > 0:
            de.sort(reverse=True)
            for k in range(len(de)):
                data = np.delete(data, de[k], axis=0)
        a = len(data)
        if i >= a:
            break
    return data


if __name__ == '__main__':
    start = time.time()
    set0 = np.loadtxt('Polyline_PCB02.txt', delimiter=',')
    set1 = np.loadtxt('PCB_c1_z5_t20.txt', delimiter='\t')
    data = np.loadtxt('convexpolygon_dlg.txt', delimiter='\t')
    # data=np.loadtxt('convexpolygon_dopp.txt',  delimiter='\t')
    print(len(data))
    plt.figure(figsize=(16, 9))
    plt.scatter(set1[:, 0], set1[:, 1], c='orange')
    plt.scatter(data[:, 0], data[:, 1], c='red')
    plt.show()
    while 1:
        max, a0, b0 = maxPoints(points=data)
        if max > 5:
            print('----')
            data = delete_linepoints(data, a0, b0, 10)
        else:
            print('终止')
            break

    print(len(data))
    plt.figure(figsize=(16, 9))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.scatter(set0[:, 0], set0[:, 1], c='yellow')
    # plt.scatter(set1[:, 0], set1[:, 1], c='orange')
    plt.scatter(data[:, 0], data[:, 1], c='r')
    # plt.scatter(data[:, 0], data[:, 1], c='b')
    plt.title('dlg_删除共线中间点-%d' % len(data))
    # plt.title('dopp_删除共线中间点-%d'%len(data))
    plt.show()
    data = delete_closepoints(data, 10)
    print(len(data))
    plt.figure(figsize=(16, 9))
    plt.scatter(set0[:, 0], set0[:, 1], c='yellow')
    # plt.scatter(set1[:, 0], set1[:, 1], c='orange')
    plt.scatter(data[:, 0], data[:, 1], c='r')
    # plt.scatter(data[:, 0], data[:, 1], c='b')

    # plt.title('dopp_删除邻近点-%d'%len(data))
    plt.title('dlg_删除邻近点-%d' % len(data))
    # plt.savefig('conpoly_dopp.png')
    plt.savefig('conpoly_dlg.png')
    plt.show()
    # np.savetxt('conpoly_dlg.txt',data,delimiter=' ')
    np.savetxt('conpoly_dopp.txt', data, delimiter=' ')
