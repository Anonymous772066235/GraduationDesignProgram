# File      :LineFit_RANSAC.py
# Author    :WJ
# Function  :
# Time      :2021/02/21
# Version   :
# Amend     :

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import time



start = time.time()


def Ransac(points,iters=1000 ,leastpoints=20, mindistance=1e-2):
    # 至少穿过leastpoints个点的直线才会被计数
    # 一个点的情况
    if len(points) == 1:
        return 1
    # 任意两个点构成一条直线，判断点集内的点是否再这条直线上（附近）
    max = leastpoints
    Am = 0
    Bm = 0
    Cm = 0
    for i in range(iters):
        # 随机选两个点去求解模型
        sample_index = random.sample(range(len(points)), 2)
        x_1 = points[sample_index[0]][0]
        y_1 = points[sample_index[0]][1]
        x_2 = points[sample_index[1]][0]
        y_2 = points[sample_index[1]][1]

        A = y_2 - y_1
        B = x_1 - x_2
        C = x_2 * y_1 - x_1 * y_2

        pointcount = 0
        if A * A + B * B > 0.01:
            for k in range(len(points)):
                if abs(A * points[k, 0] + B * points[k, 1] + C) / math.sqrt(A * A + B * B) < mindistance:
                    pointcount = pointcount + 1
        if max < pointcount:
            max = pointcount
            Am = A
            Bm = B
            Cm = C

    return np.array([Am, Bm, Cm])


def delete_points(points, Line, mindistance=1):
    A = Line[0]
    B = Line[1]
    C = Line[2]
    K = []
    for k in range(len(points)):
        if abs(A * points[k, 0] + B * points[k, 1] + C) / math.sqrt(A * A + B * B) < mindistance:
            K.append(k)
    K.sort(reverse=True)
    for i in range(len(K)):
        points = np.delete(points, K[i], axis=0)
    return points


def showLines(data, Lines, x0=0, y0=0, name='line_dlg'):
    x = np.linspace(-50, 50, 100, endpoint=True)
    x += x0
    plt.figure(figsize=(16, 9))
    plt.axis('equal')
    plt.scatter(data[:, 0], data[:, 1], c='b')
    for i in range(len(Lines)):
        A = Lines[i, 0]
        B = Lines[i, 1]
        C = Lines[i, 2]
        if abs(B) < 0.01:
            x1 = np.zeros(shape=(len(x), 1)) - C / A
            y1 = np.linspace(-20, 20, 100, endpoint=True)
            y1 += y0
            plt.plot(x1, y1, c='r')
        else:
            y = (-A * x - C) / B
            plt.plot(x, y, c='r')
    # plt.axes([-500+x0,500+x0,-500+y0,500+y0])
    plt.xlim(-200 + x0, 200 + x0)
    plt.ylim(-200 + y0, 200 + y0)

    plt.title(name)
    plt.savefig('..\\pic\\' + name + '.png')
    # plt.show()
    # plt.pause(10)
    plt.close()


def searchLines(points, num_lines,iters, leastpointsOnLine, mindistance):
    i = 0
    Lines = []
    while i < num_lines:
        i += 1
        Line = Ransac(points, iters,leastpointsOnLine, mindistance)
        Lines.append(Line)
        # points = delete_points(points, Line, mindistance + 0.5)   PCB
        # points = delete_points(points, Line, mindistance + 3)     ABC
        points = delete_points(points, Line, mindistance + 3)
    return np.array(Lines)


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


def processLines(Lines):
    for i in range(len(Lines)):
        if abs(Lines[i, 1]) > 0.01:
            Lines[i, :] = Lines[i, :] / Lines[i, 1]
    return Lines


def mergeLines_ABC(lines, X0, Y0, slope=0.1, intercept=5):
    # 根据Y轴上的截距合并
    abc = pd.DataFrame()
    abc['A'] = lines[:, 0]
    abc['B'] = lines[:, 1]
    abc['C'] = lines[:, 2]
    ABC = []
    # 先处理特殊情况，即 B=0的情况
    B0 = abc[abs(abc['B']) <= 1e-5]
    abc.drop(labels=B0.index, axis=0, inplace=True)
    abc = abc.reset_index(drop=True)
    B0['intercept'] = -np.array(B0['C']) / np.array(B0['A'])
    while len(B0) > 0:
        ac = B0[
            (B0['intercept'][0] - intercept <= B0['intercept']) & (B0['intercept'][0] + intercept >= B0['intercept'])]
        acmean = ac.mean(axis=0)
        ABC.append(np.array(acmean)[0:3])
        B0.drop(labels=ac.index, axis=0, inplace=True)
        B0 = abc.reset_index(drop=True)

    abc['a'] = -np.array(abc['A']) / np.array(abc['B'])
    abc['b'] = - (abc['A'] * X0 + np.array(abc['C'])) / np.array(abc['B'])
    abc['c'] = - (abc['A'] * X0 + np.array(abc['C'])) / np.array(abc['B'])
    while len(abc) > 0:
        a1 = abc[(abc['a'][0] - slope <= abc['a']) & (abc['a'] <= abc['a'][0] + slope) & (
                abc['b'][0] - intercept <= abc['b']) & (
                         abc['b'] <= abc['b'][0] + intercept)]
        a1mean = a1.mean(axis=0)
        ABC.append(np.array(a1mean)[0:3])
        abc.drop(labels=a1.index, axis=0, inplace=True)
        abc = abc.reset_index(drop=True)
    ABC = np.array(ABC)

    # 根据Y轴上的截距合并
    abc02 = pd.DataFrame()
    abc02['A'] = ABC[:, 0]
    abc02['B'] = ABC[:, 1]
    abc02['C'] = ABC[:, 2]

    ABC02 = []
    # 先处理特殊情况，即 B=0的情况
    B0 = abc[abs(abc['A']) <= 1e-5]
    abc02.drop(labels=B0.index, axis=0, inplace=True)
    abc02 = abc02.reset_index(drop=True)
    B0['intercept'] = -np.array(B0['C']) / np.array(B0['B'])
    while len(B0) > 0:
        ac = B0[
            (B0['intercept'][0] - intercept <= B0['intercept']) & (B0['intercept'][0] + intercept >= B0['intercept'])]
        acmean = ac.mean(axis=0)
        ABC02.append(np.array(acmean)[0:3])
        B0.drop(labels=ac.index, axis=0, inplace=True)
        B0 = abc02.reset_index(drop=True)

    abc02['a'] = -np.array(abc02['A']) / np.array(abc02['B'])
    abc02['b'] = - (abc02['B'] * Y0 + np.array(abc02['C'])) / np.array(abc02['A'])
    abc02['c'] = - (abc02['B'] * Y0 + np.array(abc02['C'])) / np.array(abc02['A'])
    while len(abc02) > 0:
        a2 = abc02[(abc02['a'][0] - slope <= abc02['a']) & (abc02['a'] <= abc02['a'][0] + slope) & (
                abc02['b'][0] - intercept <= abc02['b']) & (
                           abc02['b'] <= abc02['b'][0] + intercept)]
        a2mean = a2.mean(axis=0)
        ABC02.append(np.array(a2mean)[0:3])
        abc02.drop(labels=a2.index, axis=0, inplace=True)
        abc02 = abc02.reset_index(drop=True)

    return np.array(ABC02)


# if __name__ == '__main__':
#     # name='dopp'
#     name = 'dlg'
#     if name == 'dopp':
#         # data0 = np.loadtxt('Polyline_PCB02_500.txt', delimiter=',')
#         data0 = np.loadtxt('PCB_c0.5_z1_t10.txt', delimiter='\t')
#         data = data0[:, 0:2]
#         print(len(data))
#         data = mergeClosePoints(data, 1.8)
#         print(len(data))
#         Lines = searchLines(data, 15, 6, 0.5)
#         print(Lines)
#         print(len(Lines))
#         Lines = processLines(Lines)
#         # print(Lines)
#         # print(len(Lines))
#         Lines = mergeLines_ABC(Lines, X0=np.mean(data[:, 0]), Y0=np.mean(data[:, 1]), slope=0.1, intercept=12)
#         print(Lines)
#         print(len(Lines))
#         showLines(data, Lines, np.mean(data[:, 0]), np.mean(data[:, 1]), name='Lines_' + name + str(len(Lines)))
#         np.savetxt('Lines_' + name + str(len(Lines)) + '.txt', Lines, delimiter=' ')
#
#     elif name == 'dlg':
#         data0 = np.loadtxt('Polyline_PCB02_500.txt', delimiter=',')
#         # data0 = np.loadtxt('PCB_c0.5_z1_t10.txt', delimiter='\t')
#         data = data0[:, 0:2]
#         # print(len(data))
#         # data = mergeClosePoints(data, 1.8)
#         print(len(data))
#         Lines = searchLines(data, 16, 6, 0.5)
#         print(Lines)
#         print(len(Lines))
#         Lines = processLines(Lines)
#         Lines = mergeLines_ABC(Lines, X0=np.mean(data[:, 0]), Y0=np.mean(data[:, 1]), slope=0.1, intercept=12)
#         print(Lines)
#         print(len(Lines))
#         showLines(data, Lines, np.mean(data[:, 0]), np.mean(data[:, 1]), name='Lines_' + name + str(len(Lines)))
#         np.savetxt('Lines_' + name + str(len(Lines)) + '.txt', Lines, delimiter=' ')
#
#     TIME = time.time() - start
#     hours = TIME // 3600
#     minutes = TIME % 3600 // 60
#     seconds = TIME % 3600 % 60
#
#     print('------------------------------------------------------')
#     print('Runningtime:{:.0f} hours {:.0f} minutes {:.0f} seconds'.format(hours, minutes, seconds))
