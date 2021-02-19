# File      :LineProcess.py
# Author    :WJ
# Function  :
# Time      :2021/01/03
# Version   :
# Amend     :


import math
import numpy as np
import pandas as pd


# def ca_angle(line1=[1,1,2,2],line2=[0,0,1,1]):
#     ang1=ca_tan(line1)
#     ang2=ca_tan(line2)
#     return ang1-ang2


def ca_angle(v1, v2):
    # 计算两直线的夹角
    dx1 = v1[2] - v1[0]
    dy1 = v1[3] - v1[1]
    dx2 = v2[2] - v2[0]
    dy2 = v2[3] - v2[1]
    angle1 = math.atan2(dy1, dx1)
    angle1 = int(angle1 * 180 / math.pi)
    angle2 = math.atan2(dy2, dx2)
    angle2 = int(angle2 * 180 / math.pi)
    included_angle = angle1 - angle2
    while included_angle < -360:
        included_angle += 360
    while included_angle > 360:
        included_angle -= 360
    return included_angle


def ca_length(line1=[1, 1, 1, 2]):
    # 计算直线的长度
    len = pow(line1[0] - line1[2], 2) + pow(line1[1] - line1[3], 2)
    return math.sqrt(len)


def ca_lengthWeight(line1=[1, 1, 2, 2], line2=[0, 0, 1, 1]):
    # 计算两直线相似度权值
    len1 = ca_length(line1)
    len2 = ca_length(line2)
    max_len = max(len1, len2)
    if abs(len1 - len2) < 1e-5:
        print('两直线长度相同：')
        print(len1, len2)
        return 99
    else:
        W_kt = max_len / abs(len1 - len2)
        return W_kt


def Line_SlopeInterceptForm(line=[1, 1, 1, 2]):
    # 计算直线的斜率（以角度形式输出）、截距
    if abs(line[0] - line[2]) < 1e-5:
        a = 999.99  # 标识
        b = (line[0] + line[2]) / 2  # 此时b为直线与x轴交点
        return a, b
    else:
        tan = (line[3] - line[1]) / (line[2] - line[0])
        b = line[1] - tan * line[0]
    return tan, b


def Line_GeneralEquation(line=[1, 1, 1, 2]):
    # 一般式 Ax+By+C=0
    A = line[3] - line[1]
    B = line[0] - line[2]
    C = line[2] * line[1] - line[0] * line[3]
    return A, B, C


def mergeLines_ABC(lines, slope=0.1, intercept=5):
    abc = pd.DataFrame()
    a = []
    b = []
    c = []
    for i in range(len(lines)):
        ai, bi, ci = Line_GeneralEquation(lines[i, :])
        a.append(ai)
        b.append(bi)
        c.append(ci)
    abc['A'] = a
    abc['B'] = b
    abc['C'] = c
    print(abc)
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
    abc['b'] = np.array(abc['C']) / np.array(abc['B'])
    while len(abc) > 0:
        a1 = abc[(abc['a'][0] - slope <= abc['a']) & (abc['a'] <= abc['a'][0] + slope) & (
                    abc['b'][0] - intercept <= abc['b']) & (
                         abc['b'] <= abc['b'][0] + intercept)]
        a1mean = a1.mean(axis=0)
        ABC.append(np.array(a1mean)[0:3])
        abc.drop(labels=a1.index, axis=0, inplace=True)
        abc = abc.reset_index(drop=True)

    return np.array(ABC)


def mergeLines_ab(lines, a0=0.1, b0=5):
    ab = pd.DataFrame()
    a = []
    b = []
    for i in range(len(lines)):
        ai, bi, ci = Line_SlopeInterceptForm(lines[i, :])
        a.append(ai)
        b.append(bi)
    ab['a'] = a
    ab['b'] = b
    AB = []
    while len(ab) > 0:
        a1 = ab[(ab['a'][0] - a0 <= ab['a']) & (ab['a'] <= ab['a'][0] + a0) & (ab['b'][0] - b0 <= ab['b']) & (
                ab['b'] <= ab['b'][0] + b0)]
        a1mean = a1.mean(axis=0)
        AB.append(np.array(a1mean))
        ab.drop(labels=a1.index, axis=0, inplace=True)
        ab = ab.reset_index(drop=True)
    return np.array(AB)


def GetIntersectPointofLines(lines):
    P_x = []
    P_y = []
    for i in range(len(lines)):
        A1, B1, C1 = lines[i, 0], lines[i, 1], lines[i, 2]
        for j in range(i, len(lines)):
            A2, B2, C2 = lines[j, 0], lines[j, 1], lines[j, 2]
            m = A1 * B2 - A2 * B1
            if abs(m)>0.1:
                x = (C2 * B1 - C1 * B2) / m
                y = (C1 * A2 - C2 * A1) / m
                P_x.append(x)
                P_y.append(y)
    P=pd.DataFrame()
    P['X']=P_x
    P['Y']=P_y
    return np.array(P)


if __name__ == '__main__':
    lines = np.array([[0, 0, 1, 1],
                      [0, 0, 2, 2],
                      [-1, 5, 4, 9.9],
                      [0, 0, 2, 1],
                      [0, 5, 2, 1],
                      [0, 0, 3, 3.1],
                      [0, 5, 4, 2]])
    ABC = mergeLines_ABC(lines)
    print(ABC)
    points=GetIntersectPointofLines(ABC)
    print(points)
