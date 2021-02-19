# File      :TransformationMatrix.py
# Author    :WJ
# Function  :
# Time      :2021/01/30
# Version   :
# Amend     :
import numpy as np
import matplotlib.pyplot as plt
import Visualization as Vs


def ca_rt(DOPP, DLG, filename='图匹配所求参数.txt'):
    import math
    import numpy as np
    # print('\n求参：\n')
    Y = np.matrix(DLG)
    X = np.matrix(DOPP)

    mu_y = np.mean(Y, axis=0)
    mu_x = np.mean(X, axis=0)
    # print('点集Y的重心化坐标：')
    # print(mu_y.round(3))
    # print('点集X的重心化坐标：')
    # print(mu_x.round(3))

    for i in range(len(Y)):  # 重心化
        Y[i, :] -= mu_y
    for i in range(len(X)):
        X[i, :] -= mu_x

    a = Y
    b = X

    for i in range(len(Y)):
        sin = (a[i, 0] * b[i, 1] - a[i, 1] * b[i, 0])
        cos = (a[i, 0] * b[i, 0] + a[i, 1] * b[i, 1])
    phi = math.atan(sin / cos)

    # print('\n旋转角phi=')
    # print(phi)

    R = np.matrix([[math.cos(phi), math.sin(phi)],
                   [-math.sin(phi), math.cos(phi)]])

    # print('\n旋转矩阵R=')
    # print( R)

    T = np.matrix(mu_y.transpose() - np.dot(R, mu_x.transpose()))

    # print('\n平移矩阵T=')
    # print(T)

    tf = np.zeros((3, 3))
    tf[0, 0] = R[0, 0]
    tf[0, 1] = R[0, 1]
    tf[1, 0] = R[1, 0]
    tf[1, 1] = R[1, 1]
    tf[0, 2] = T[0, 0]
    tf[1, 2] = R[1, 0]

    with open('..\\data\\' + filename, 'w') as f:  # 输出参数
        f.write('###该文件为变换参数及矩阵输出文件###\n')
        f.write(' \n')
        f.write('旋转角phi=\n')
        f.write(str(phi))
        f.write('\n \n')
        f.write('旋转矩阵R=\n')
        f.write(str(R))
        f.write('\n \n')
        f.write('平移矩阵TR=\n')
        f.write(str(T))
        f.write('\n \n')
        f.write('总变换矩阵tf=\n')
        f.write(str(tf))
    return R, T


def transformation(DOPP, R, T, filename='DOPP变换后.txt'):
    import numpy as np
    dopp_t = np.matrix(np.dot(R, DOPP.transpose()) + T)
    dopp = dopp_t.transpose()
    np.savetxt('..\\data\\' + filename, dopp, fmt="%5.3f", delimiter=" ")
    return np.array(dopp)


if __name__ == '__main__':
    DOPP = np.loadtxt('CrossPoints_DOPP_OK.txt', delimiter=' ')
    DLG = np.loadtxt('CrossPoints_DLG_OK.txt', delimiter=' ')
    R, T = ca_rt(DOPP, DLG, '直线交点图匹配所求参数.txt')
    DOPP2 = transformation(DOPP, R, T)
    plt.figure(figsize=(16, 9))
    Vs.VisualizePoints(DLG, 'r', 'DLG')
    Vs.VisualizePoints(DOPP2, 'b', 'DOPP')
    plt.legend(loc='best')
    plt.savefig('直线交点初始匹配.png')
    plt.show()
