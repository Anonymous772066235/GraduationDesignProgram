# File      :Registratioln_H.py
# Author    :WJ
# Function  :   在高程方向配准dlg与lidar点云：先获取lidar点云的屋顶点云，再与建筑物屋顶dlg配准
# Time      :2021/03/13
# Version   :
# Amend     :

import numpy as np
from scipy.spatial import KDTree


def roof(data, h=0.5):
    max = np.max(data[:, 2])
    # print(max)
    b = [x > max - h for x in data[:, 2]]
    data = data[b, :]
    # print(data)
    # print(len(data))
    return data


def registration_roof(target, data0, r=0.1, hh=0.5):
    data = roof(data0, 1)
    tree = KDTree(data[:, :2], 10)
    h = []
    for i in range(len(target)):
        idx1 = tree.query(target[i, :2], k=1, distance_upper_bound=r)
        if idx1[0] < r:
            h.append(target[i, 2] - data[idx1[1], 2])
    h = np.array(h)

    min = np.min(h)
    b = [x < min + hh for x in h]
    h = h[b]
    mean = np.mean(h)
    data0[:, 2] += mean
    mse = 0
    for i, x in enumerate(h):
        mse += abs(x - mean)
    z_rmse = mse / len(h)
    return data0,mean,z_rmse


def registration_vertialPoints(target, data0, r=0.1):
    data = data0
    tree = KDTree(data[:, :2], 10)
    h = []
    for i in range(len(target)):
        idx1 = tree.query(target[i, :2], k=1, distance_upper_bound=r)
        if idx1[0] < r:
            h.append(target[i, 2] - data[idx1[1], 2])
    h = np.array(h)

    while abs(np.min(h)-np.mean(h))>0.5:
        h=np.delete(h,np.argmin(h))
    while abs(np.max(h)-np.mean(h))>0.5:
        h=np.delete(h,np.argmax(h))
    mean = np.mean(h)

    data0[:, 2] += mean
    mse=0
    for i,x in enumerate(h):
        mse+=abs(x-mean)
    z_rmse = mse/len(h)
    return data0,mean ,z_rmse

if __name__ == "__main__":
    lidar0 = np.loadtxt('..\\data\\CSU ABC_2D_cut.txt', delimiter=',')
    lidar = lidar0[:, :3]

    dlg0 = np.loadtxt('..\\data\\Polyline_ABC.txt', delimiter=',')
    dlg = dlg0[:, :3]

    lidar_h = registration_roof(dlg, lidar)
    np.savetxt('..\\data\\CSU ABC_h.txt', lidar_h, delimiter=',')

'''
#高程精度评定
#2020/08/18
import numpy as np
from scipy.spatial import KDTree
print('导入点云数据：')
data=np.loadtxt('./ICP输出/ACB02/CSU ABC-三维配准.txt', delimiter=' ')      #导入
data=data[:,0:3]
data2=data[:,0:2]
N=len(data)
print(N)
T=KDTree(data2,10)                   #构建KD树
print('导入高程点：')
points=np.loadtxt('./ICP输出/ACB02/ACB高程模型集-2000.txt')
points=points[:,0:3]
points2=points[:,0:2]
B=[]
R=1
for i in range(len(points[:,0])):
    idx1=T.query(points2[i,:],k=1,distance_upper_bound=R)
    B.append((idx1[1]))

print(B)
r=[]
for i in range(len(B)):
    if B[i]==N:
        r.append(i)
print(r)
r.sort(reverse=True)
print(r)
for i in range(len(r)):
    del B[r[i]]
d=[]
print(B)
rr=[]
for i in range(len(B)):
    d.append(points[i,2]-data[B[i],2])

for i in range(len(d)):
    if abs(d[i])>=0.5:
        rr.append(i)

rr.sort(reverse=True)
print(rr)
for i in range(len(rr)):
    del d[rr[i]]
print('对每个高程点，%sm范围内地面点的高程精度分别为'%R)
print(d)
print('\n高程精度即符合条件的高程点的与地面点平均高差为：')
print(np.mean(d))

'''
