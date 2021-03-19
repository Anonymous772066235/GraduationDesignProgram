# File      :ICP_Encapsulation.py
# Author    :WJ
# Function  :
# Time      :2021/03/04
# Version   :3.0
# Amend     :11/16    将原函数拆分为两个函数，以提高代码可移植性：
#                     (1) ICP负责输出仿射ICP匹配过程中的仿射变换参数；
#                     (2) 负责用ICP输出的仿射变换参数对点集进行仿射变换。
#            11/24    debug:
#                     (1)增加新的else分支
#                     (2)设置新的终止条件：delta相差小于1e-10时视为收敛
#                     (3)增加显示重叠度功能
#       2021/01/24    debug:
#                     (1)在封装好的仿射ICP基础上改为刚体ICP
#       2021/03/19    增加精度评定功能：
#                     (1)将rmse终止判断功能结合精度评定功能，移到求最近点集方法中



def ICP(rowdata, target, MaxDistance=50, MaxIteration=200):
    # 参数一位待匹配的二维点云，参数二为目标点云，参数三为最近点集搜索半径，参数四为最大迭代次数
    dopp = rowdata
    dlg = target

    class ICP:
        import numpy as np
        dlg = np.array(0)  # 模型集
        dopp = np.array(0)  # 初始数据集
        Y = []  # 最近点集
        X = []
        RMSE = 0  # 均方根误差（root mean squared deltaor）描述数据集到最近点集距离
        delta = 0  # 均方根误差较差：RMSE-RMSE_pre
        R = [[1, 0],
             [0, 1]]
        T = [[0],
             [0]]
        N = 0
        phi = 0
        J = 0

        def __init__(self, dopp, dlg):
            ICP.dopp = dopp
            ICP.dlg = dlg
            print('开始ICP匹配')
            print('----------------------------------')

        def __del__(self):
            print('----------------------------------')
            print('ICP匹配完成\n')

        def ca_Y(self, MaxDistance=20):  # 求最近点集Y MaxDistanc为搜索最邻近点时的范围
            import numpy as np
            from scipy.spatial import KDTree
            y = []
            mse = []
            XX = []
            P = ICP.dlg
            X = ICP.dopp

            Tree = KDTree(P, 10)  # 建立KDTree
            for i in range(X.shape[0]):
                idx1 = Tree.query(X[i, :], k=1, distance_upper_bound=MaxDistance)
                if idx1[0] < MaxDistance:
                    mse.append(idx1[0])
                    XX.append(X[i])
                    y.append(P[idx1[1]])
            ICP.X = np.array(XX)
            ICP.Y = np.array(y)
            ICP.N = len(ICP.Y)
            rmse = np.mean(mse)
            print('rmse:\t', rmse)
            delta = rmse - ICP.RMSE
            print('delta:\t', delta)
            if (abs(delta) < 1e-10):
                ICP.J = -1
                print('01迭代收敛！')
            else:
                ICP.RMSE = rmse
                ICP.delta = delta
            # print('|满足要求的最近点集与数据点集重叠度：\t%2.3f' % ((len(y) / ICP.dopp.shape[0]) * 100), '\t%|')
            # print('|满足要求的最近点集与目标点集重叠度：\t%2.3f' % ((len(y) / ICP.dlg.shape[0]) * 100), '\t%|')

        def ca_RT(self):
            import math
            import numpy as np
            mu_y = np.mean(ICP.Y, axis=0)
            mu_x = np.mean(ICP.X, axis=0)
            Y = np.array(ICP.Y)
            X = np.array(ICP.X)

            for i in range(ICP.N):
                Y[i, :] -= mu_y
            for i in range(ICP.N):
                X[i, :] -= mu_x
            b = Y
            a = X
            sin = 0
            cos = 0
            # print('\n计算旋转角phi:')
            for i in range(ICP.N):
                sin = sin - (a[i, 0] * b[i, 1] - a[i, 1] * b[i, 0])
                cos = cos + (a[i, 0] * b[i, 0] + a[i, 1] * b[i, 1])
            phi = math.atan(sin / cos)

            R = np.matrix([[math.cos(phi), math.sin(phi)],
                           [-math.sin(phi), math.cos(phi)]])

            # print('\n计算平移矩阵：')
            T = np.matrix(mu_y.transpose() - np.dot(R, mu_x.transpose()))
            T = T.transpose()

            if (R == ICP.R).all():
                if (T == ICP.T).all():
                    ICP.J = -1
                    print('00迭代收敛！')
                else:  # 11/24 debug新增分支
                    tem = np.array(np.dot(R, ICP.dopp.transpose()) + T).transpose()
                    if (ICP.dopp == tem).all():
                        ICP.J = -1
                        print('02迭代收敛！')
                    else:
                        ICP.dopp = tem
                        ICP.T = R * ICP.T + T
                        ICP.phi = ICP.phi + phi
                        R2 = np.array([[math.cos(ICP.phi), math.sin(ICP.phi)],
                                        [-math.sin(ICP.phi), math.cos(ICP.phi)]])
                        ICP.R = R2

            else:
                tem = np.array(np.dot(R, ICP.dopp.transpose()) + T).transpose()
                if (ICP.dopp == tem).all():
                    ICP.J = -1
                    print('02迭代收敛！')
                else:
                    ICP.dopp = tem
                    ICP.phi = ICP.phi + phi
                    R2 = np.array([[math.cos(ICP.phi), math.sin(ICP.phi)],
                                    [-math.sin(ICP.phi), math.cos(ICP.phi)]])
                    ICP.T = R * ICP.T + T
                    ICP.R = R2

    import time
    start = time.clock()
    A = ICP(dopp, dlg)
    print("正在进行第 \t1 次匹配。")
    A.ca_Y(MaxDistance)
    A.ca_RT()
    # 迭代
    i = 1
    while (MaxIteration - i > 0):
        i += 1
        print('----------------------------------')
        print("正在进行第 \t%d 次匹配。" % i)
        if MaxDistance > 20:
            MaxDistance -= 5.
        elif MaxDistance > 2:
            MaxDistance -= 2
        elif MaxDistance > 0.50:
            MaxDistance -= 0.1
        A.ca_Y(MaxDistance)
        A.ca_RT()
        if A.J != 0:
            break
    if A.J == 0:
        print('迭代未收敛。')
        print('\n总匹配次数为\t%d次。' % i)
    else:
        print('\n总匹配次数为\t%d次。' % (i - 1))
    end = time.clock()
    print('Running time: %s Seconds\t' % (end - start))
    return A.R, A.T, A.RMSE


def Transform(data, R, T):
    import numpy as np
    data_T = data.transpose()
    # print("开始刚体变换：")
    data_T = np.array(np.dot(R, data_T) + T)
    data = data_T.transpose()
    # print("刚体变换完成。")
    return data


# 测试-----------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    import numpy as np

    # data1 = np.loadtxt('..\\data\\ABC\\dopp_transformed_abc.txt', delimiter=' ')
    data1 = np.loadtxt('..\\data\\ABC\\dopp_transformed_ransac_abc.txt', delimiter=' ')
    dopp = np.array(data1[:, 0:2])
    data2 = np.loadtxt('..\\data\\ABC\\Polyline_ABC.txt', delimiter=',')
    dlg = np.array(data2[:, 0:2])

    # 以上为数据集准备，以下为函数调用
    R, T = ICP(dopp, dlg, 45, 100)  # ***
    print(R)
    print(T)
    dopp_TF = Transform(dopp, R, T)
    np.savetxt("..\\data\\ABC\\dopp_TF_ransac_abc.txt", dopp_TF, delimiter=',')
    # 测试-----------------------------------------------------------------------------------------------------------
