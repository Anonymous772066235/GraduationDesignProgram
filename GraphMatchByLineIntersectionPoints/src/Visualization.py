# File      :Visualization.py
# Author    :WJ
# Function  :
# Time      :2021/01/14
# Version   :
# Amend     :


import matplotlib.pyplot as plt


def VisualizeMacthLine(P_dlg, P_dopp, row_ind, col_ind):
    for i in range(len(row_ind)):
        X = [P_dlg[row_ind, 0], P_dopp[col_ind, 0]]
        Y = [P_dlg[row_ind, 1], P_dopp[col_ind, 1]]
        plt.plot(X, Y, color='b')


def VisualizePoints(points, color, label):
    plt.scatter(points[:, 0], points[:, 1], color=color, label=label)


def VisualizeConPolyPoints(data, P, pic='pic'):
    plt.figure(figsize=(16, 9))
    plt.axis('equal')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.scatter(data[:, 0], data[:, 1], c='blue', marker='.')
    plt.scatter(P[:, 0], P[:, 1], c='red')
    plt.title(pic + '-%d' % len(P))
    plt.savefig('..\\pic\\conpoly_points_' + pic + '.png')
    plt.close()


def Visualize2PointClouds(cloud1,cloud2, pic='pic',feature1=['blue','cloud1','.'],feature2=['red','cloud2','.']):
    plt.figure(figsize=(16, 9))
    plt.axis('equal')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.scatter(cloud1[:, 0], cloud1[:, 1], c=feature1[0],label=feature1[1] ,marker=feature1[2])
    plt.scatter(cloud2[:, 0], cloud2[:, 1], c=feature2[0],label=feature2[1] ,marker=feature2[2])
    plt.title(pic)
    plt.savefig('..\\pic\\' + pic + '.png')
    plt.close()

def VisualizeMatch(P_dopp, P_dlg, row, col, pic='凸多边形顶点'):
    plt.figure(figsize=(12, 9))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    VisualizePoints(P_dlg, color='red', label='dlg_points')
    VisualizePoints(P_dopp, color='green', label='dopp_points')
    VisualizeMacthLine(P_dlg, P_dopp, row, col)
    plt.axis('equal')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Match_' + pic)
    plt.legend(loc='best')
    plt.savefig('..\\pic\\Match_' + pic + '.png', dpi=300)
