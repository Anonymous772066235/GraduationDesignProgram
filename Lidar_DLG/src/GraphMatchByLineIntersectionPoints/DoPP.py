# File      :DoPP.py
# Author    :WJ
# Function  :
# Time      :2021/01/22
# Version   :
# Amend     :
import numpy as np
import matplotlib.pyplot as plt
#类设计之DoPP
#2020/08/02修正版
#修正Delete方法 增加ca_DATA3、ShowPointCould方法
#解决三维可视化点云时出现的内存不足的问题、简化可视化代码
class DoPP:
    DATA=[]
    DATA2=[]
    DATA3=[]
    MAX=[]
    MIN=[]
    max=[]
    min=[]
    C=1
    T=20
    W=0
    H=0
    GRID=[]
    Z=0


    def __init__(self):
        print('程序开始执行\n')
    def __del__(self):
        print('程序执行完毕\n')

    def InputData(self,filename,de=','):
        print('开始导入数据：')

        import numpy as np
        data1 = np.loadtxt(filename, delimiter=de)  # delimiter参数依据原始文本数据每行数字之间符号，这里为\t
        data1 = data1[:, 0:4]
        for i in range(len(data1)):
            data1[i,3]=i
        print(len(data1),len(data1[1]))
        DoPP.DATA=data1
        del data1
        np.savetxt("DATA.txt", DoPP.DATA, fmt="%5.3f", delimiter="\t", )  # 将读取的文件保存到另一文本
        print('导入数据成功！')

    def PrintData(self):
        print('开始打印DATA:')
        print(DoPP.DATA)
        print('完成打印！')


    def SerachMaxMin(self):
        print('开始搜索极值：')
        import numpy as np
        for i in range(3):
            DoPP.MAX.append(np.max(DoPP.DATA[:,i]))
            DoPP.MIN.append(np.min(DoPP.DATA[:,i]))
        print(DoPP.MAX)
        print(DoPP.MIN)
        print('极值搜索完毕！')

    def Grid(self,C):
        print('开始划分格网：')
        DoPP.C=C
        import math
        for i in range(len(DoPP.MAX)):
            DoPP.max.append(math.ceil(DoPP.MAX[i]))
            DoPP.min.append(math.floor(DoPP.MIN[i]))
        DoPP.W=int((DoPP.max[0]-DoPP.min[0])/C)
        DoPP.H=int((DoPP.max[1]-DoPP.min[1])/C)
        print(DoPP.max)
        print(DoPP.min)
        print(DoPP.W,DoPP.H)
        print('格网划分完成！')

    def Ca_DoPP(self,T):
        DoPP.T=T
        import numpy as np
        import math
        print('开始DoPP计算：')
        Grid=[]
        Grid=np.zeros((int(DoPP.W+1),int(DoPP.H+1)))
        for i in range (len(DoPP.DATA2)):
            a=math.floor((DoPP.DATA2[i,0]-DoPP.min[0])/DoPP.C)
            b=math.floor((DoPP.DATA2[i,1]-DoPP.min[1])/DoPP.C)
            Grid[a,b]+=1
        np.savetxt("Grid1.txt", Grid, fmt="%2.0f", delimiter="\t", )
        for i in range(DoPP.W):
            for j in range(DoPP.H):
                if Grid[i,j]<T:
                    Grid[i,j]=0
        DoPP.GRID = Grid
        del Grid
        np.savetxt("Grid2.txt", DoPP.GRID, fmt="%2.0f", delimiter="\t", )
        print('计算DoPP成功！')


    def Delete(self,Z):
        DoPP.Z=Z
        import numpy as np
        import os
        print('开始删除非地面点：')
        DoPP.DATA2=[]
        for j in range(len(DoPP.DATA)):
            if(DoPP.DATA[j,2]>=Z):
                DoPP.DATA2.append(DoPP.DATA[j,:])
        np.savetxt("DATA2.txt", DoPP.DATA2, fmt="%5.3f", delimiter="\t", )  # 将读取的文件保存到另一文本
        DoPP.DATA2 = np.loadtxt('DATA2.txt', delimiter='\t')

        print('删除工作完成！')



    def ShowAll(self):
        import numpy as np
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        # 列表解析x,y,z的坐标
        print('开始绘制输出三维点云：')
        x = DoPP.DATA[:,0]
        y = DoPP.DATA[:,1]
        z = DoPP.DATA[:,2]
        # 开始绘图
        fig = plt.figure(dpi=150)
        ax = fig.add_subplot(111, projection='3d')
        # 标题
        plt.title('3D point cloud')
        # 利用xyz的值，生成每个点的相应坐标（x,y,z）
        ax.scatter(x, y, z, c='b', marker='.', s=2, linewidth=0, alpha=1, cmap='spectral')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        # 显示
        plt.show()
        print('图像绘制完成！')

    def Show_Point(self):
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.pyplot import MultipleLocator
        from mpl_toolkits.mplot3d import Axes3D
        print('开始绘制输出二维非地面点云：')
        # 列表解析x,y,z的坐标
        X=DoPP.DATA2[:,0]
        Y=DoPP.DATA2[:,1]
        plt.figure(dpi=150)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title('2D point cloud')
        plt.plot(Y, X, c='r', marker='.', linestyle='None', markersize=0.05) #ls=''
        plt.show()
        print('图像绘制完成！')

    def Show_GRID(self):
        import matplotlib.pyplot as plt
        import numpy as np
        print('开始绘制GRID：')
        plt.figure(dpi=150)
        plt.gca().set_aspect('equal')
        plt.title('GRID')
        plt.imshow(DoPP.GRID)
        plt.savefig('grid.png')
        # plt.show()
        print(DoPP.GRID)
        print('图像绘制完成！')

    def ca_DATA3(self,filename='data3.txt'):
        import math
        import numpy as np
        print("开始统计GRID内点云数据：")
        DATA3=[]
        for i in range(len(DoPP.DATA2)):
            a = math.floor((DoPP.DATA2[i, 0] - DoPP.min[0]) / DoPP.C)
            b = math.floor((DoPP.DATA2[i, 1] - DoPP.min[1]) / DoPP.C)
            if (DoPP.GRID[a, b]>0.0):
                DATA3.append(DoPP.DATA2[i,:])
        np.savetxt(filename, DATA3, fmt="%5.3f", delimiter="\t", )
        DoPP.DATA3=np.loadtxt(filename, delimiter='\t')

        print("GRID内点云数据统计成功！")

    def ShowPointCould(self):
        import numpy as np
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        # 列表解析x,y,z的坐标
        print('开始绘制输出建筑物立面三维点云：')
        x = DoPP.DATA3[:, 0]
        y = DoPP.DATA3[:, 1]
        z = DoPP.DATA3[:, 2]
        # 开始绘图
        fig = plt.figure(dpi=150)
        ax = fig.add_subplot(111, projection='3d')
        # 标题
        plt.title('3D point cloud')
        # 利用xyz的值，生成每个点的相应坐标（x,y,z）
        ax.scatter(x, y, z, c='b', marker='.', s=2, linewidth=0, alpha=1, cmap='spectral')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        # 显示
        plt.show()
        print('图像绘制完成！')

def plane(data,low=0,high=1):
    Plane=[]
    for i in range(len(data)):
        if   low<data[i,2]<high:
            Plane.append(data[i,:])
    return np.array(Plane)




#类结束



import time
import matplotlib.pyplot as plt
start =time.clock()
A=DoPP()
# A.InputData('CSU_PCB sample 0.8_cut.txt')
A.InputData('..\\data\\CSU ABC sample 0.5_cut.txt')
A.PrintData()
#A.ShowAll()
A.Delete(5)
#A.Show_Point()
A.PrintData()
A.SerachMaxMin()
A.Grid(1)
A.Ca_DoPP(5)
A.Show_GRID()
A.ca_DATA3('..\\data\\ABC_c1_z5_t5.txt')

plt.figure(figsize=(16,9))
plt.axis('equal')
plt.scatter(A.DATA3[:,0],A.DATA3[:,1],c='black',s=1)
plt.axis('off')
plt.xticks([])
plt.yticks([])
plt.savefig('..\\data\\dopp_abc_result__c1_z5_t5.png')


end = time.clock()
print('Running time: %s Seconds'%(end-start))