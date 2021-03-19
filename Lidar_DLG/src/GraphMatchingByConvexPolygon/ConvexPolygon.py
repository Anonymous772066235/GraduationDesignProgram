# File      :ConvexPolygon.py
# Author    :WJ
# Function  :
# Time      :2021/01/20
# Version   :
# Amend     :
# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import ConvexHull
import cv2 as cv


def ConvexPolygon(set):
    hull = ConvexHull(set)
    b = []
    # plt.plot(set[:, 0], set[:, 1], 'o')
    for simplex in hull.simplices:
        for i in range(len(simplex)):
            b.append(simplex[i])
    b = np.unique(b)

    return b


def minRect(points):
    rect = cv.minAreaRect(points)  # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
    box = cv.boxPoints(rect)
    return box


def delPointOnLine(points):
    pass


if __name__ == '__main__':
    set0 = np.loadtxt('Polyline_PCB02.txt', delimiter=',')
    set1 = np.loadtxt('PCB_c1_z5_t20.txt', delimiter='\t')
    set0 = set0[:, 0:2]
    set1 = set1[:, 0:2]
    b1 = ConvexPolygon(set0)
    print(b1)
    b2 = ConvexPolygon(set1)
    print(b2)

    a = np.array(set0[b1, :], dtype=np.float32)
    a2 = np.array(set1[b2, :], dtype=np.float32)

    print(a, type(a), a.shape)
    print(a2, type(a2), a2.shape)
    plt.figure(figsize=(16, 9))
    plt.scatter(set0[:, 0], set0[:, 1], c='yellow')
    plt.scatter(a[:, 0], a[:, 1], c='red')
    plt.title('dlg-%d' % len(a))
    plt.show()
    plt.figure(figsize=(16, 9))
    plt.scatter(set1[:, 0], set1[:, 1], c='orange')
    plt.scatter(a2[:, 0], a2[:, 1], c='blue')
    plt.title('dopp-%d' % len(a2))
    plt.show()
    np.savetxt('convexpolygon_dopp.txt', a2, delimiter='\t')
    np.savetxt('convexpolygon_dlg.txt', a, delimiter='\t')


