# File      :HierarchicalClustering.py
# Author    :WJ
# Function  :层次聚类
# Time      :2021/02/07
# Version   :
# Amend     :

from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle  ##python自带的迭代器模块


def mergeClosePoints(Points, distance=0.5):
    xy = pd.DataFrame()
    x = []
    y = []
    for i in range(len(Points)):
        xi = Points[i, 0]
        yi = Points[i, 1]
        x.append(xi)
        y.append(yi)
    xy['x'] = x
    xy['y'] = y
    XY = []

    while len(xy) > 0:
        a1 = xy[(xy['x'][0] - distance <= xy['x']) & (xy['x'] <= xy['x'][0] + distance) & (
                xy['y'][0] - distance <= xy['y']) & (
                        xy['y'] <= xy['y'][0] + distance)]
        a1mean = a1.mean(axis=0)
        XY.append(np.array(a1mean))
        xy.drop(labels=a1.index, axis=0, inplace=True)
        xy = xy.reset_index(drop=True)

    return np.array(XY)


def HierarchicalClustering(points, k=1):
    if len(points) > 1000:
        points = mergeClosePoints(points, 1.5)

    # 设置聚类数
    n_clusters_ = k

    ##设置分层聚类函数
    linkages = ['ward', 'average', 'complete']
    ac = AgglomerativeClustering(linkage=linkages[2], n_clusters=n_clusters_)

    ##训练数据
    ac.fit(points)

    ##每个数据的分类
    lables = ac.labels_

    # ##绘图
    # plt.figure(figsize=(16, 9))
    # plt.axis('equal')
    # colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    # for k, col in zip(range(n_clusters_), colors):
    #     ##根据lables中的值是否等于k，重新组成一个True、False的数组
    #     my_members = lables == k
    #     ##X[my_members, 0] 取出my_members对应位置为True的值的横坐标
    #     plt.plot(points[my_members, 0], points[my_members, 1], col + '.')
    #
    # plt.title(picname+' clusters: %d' % n_clusters_)
    # plt.savefig('.\\pic\\'+picname+'\\HierarchicalClustering_'+picname+'.png')#..\\..\\pic\\ABC\\
    # plt.close()
    return points,lables


if __name__ == '__main__':
    # data0=np.loadtxt('..\\data\\Polyline_PCB02_500.txt',delimiter=',')
    data0 = np.loadtxt('..\\data\\PCB_c1_z5_t20.txt', delimiter='\t')
    data = data0[:, 0:2]
    HierarchicalClustering(data, 2,'dopp')
