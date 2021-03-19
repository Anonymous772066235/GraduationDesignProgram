# File      :TransformationMatrix.py
# Author    :WJ
# Function  :
# Time      :2021/03/19
# Version   :
# Amend     :
import numpy as np


def ca_rt(DOPP, DLG):
    import math
    import numpy as np
    Y = np.matrix(DLG)
    X = np.matrix(DOPP)
    mu_y = np.mean(Y, axis=0)
    mu_x = np.mean(X, axis=0)
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
    R = np.matrix([[math.cos(phi), math.sin(phi)],
                   [-math.sin(phi), math.cos(phi)]])
    T = np.matrix(mu_y.transpose() - np.dot(R, mu_x.transpose()))
    return R, T


def transformation(DOPP, R, T):
    import numpy as np
    dopp_t = np.matrix(np.dot(R, DOPP.transpose()) + T)
    dopp = dopp_t.transpose()
    # np.savetxt('..\\data\\' + filename, dopp, fmt="%5.3f", delimiter=" ")
    return np.array(dopp)


def r1t1r2t2z(R1,T1,R2,T2,Z):
    R=np.dot(R2,R1)
    T=np.dot(R2,T1)+T2
    TF = np.eye(4)
    TF[:2,:2]=R
    TF[:2,3:4]=T
    TF[2,3]=Z
    return TF


def output(R1,T1,R2,T2,Z,rmse,z_rmse,path):
    TF=r1t1r2t2z(R1, T1, R2, T2, Z)
    with open(path,'w') as f:
        f.write('Header----------------------------------------------------\n')
        f.write('该文件为Lidar_DLG项目所求配准参数及精度输出文件\n')
        f.write('Total---------------------------------------------\n')
        f.write('总配准参数矩阵TF:\n')
        f.write(str(TF))
        f.write('\n总配准精度：       %2.3f\n'%(np.sqrt(rmse*rmse+z_rmse*z_rmse)))
        f.write('Detial--------------------------------------------\n')
        f.write('图匹配所求旋转矩阵R1:\n')
        f.write(str(R1))
        f.write('\n图匹配所求平移矩阵T1:\n')
        f.write(str(T1))
        f.write('\nICP匹配所求旋转矩阵R2:\n')
        f.write(str(R2))
        f.write('\nICP匹配所求平移矩阵T2:\n')
        f.write(str(T2))
        f.write('\n平面配准精度rmse:  %2.3f\n'%rmse)
        f.write('高程配准精度z_rmse:%2.3f\n'%z_rmse)
        f.write('Footer----------------------------------------------------')
        f.close()


if __name__ == '__main__':
    r1=[[ 0.9971231,  -0.07579923],
        [ 0.07579923,  0.9971231 ]]
    t1=[4072.8404043 ,
        1713.70722919]
    r2=[[ 0.99986995 , 0.01612704],
         [-0.01612704 , 0.99986995]]
    t2=[-28.28831309,
         64.13507406]
    z=42.91075
    tf=r1t1r2t2z(r1,t1,r2,t2,z)
    print(tf)