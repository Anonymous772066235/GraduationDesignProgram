# File      :main_LineIntersectionPoints.py
# Author    :WJ
# Function  :
# Time      :2021/03/18
# Version   :
# Amend     :

import numpy as np
import src.DensityOfProjectedPoints as DoPP
import src.GraphMatchByLineIntersectionPoints as GM_LIPoint
import src.IterativeClosestPoints as ICP
import src.Registration_H as Reg_H

if __name__ == '__main__':
    # name = 'PCB'
    name = 'ABC'
    if name == 'ABC':

        # 1.导入原始数据
        lidar0 = np.loadtxt('..\\data\\ABC\\CSU_ABC_cut.txt', delimiter=',')
        lidar = lidar0[:, :3]
        dlg0 = np.loadtxt('..\\data\\ABC\\Polyline_ABC.txt', delimiter=',')
        dlg = dlg0[:, :3]

        # 2.DoPP
        lidar_dopp = DoPP.DoPP.run(lidar, 10, 1, 10, name)

        # 3.图匹配
        R1, T1, lidar_dopp = GM_LIPoint.all_Ransac.run(dlg, lidar_dopp, name)

        # 4.ICP
        R2, T2, rmse = ICP.ICP(lidar_dopp, dlg[:, :2], 45, 100)

        # 5.应用变换参数、输出平面配准完成的LiDARD点云
        lidar[:, :2] = ICP.Transform(lidar[:, :2], R1, T1)
        lidar[:, :2] = ICP.Transform(lidar[:, :2], R2, T2)
        np.savetxt('..\\output\\ABC\\CSU_ABC_2D_LIPoint.txt', lidar, delimiter=',')

        # 6.高程配准、输出配准好的LiDAR点云
        lidar, Z, z_rmse = Reg_H.registration_roof(dlg, lidar)
        np.savetxt('..\\output\\ABC\\CSU_ABC_3D_LIPoint.txt', lidar, delimiter=',')
        GM_LIPoint.output(R1, T1, R2, T2, Z, rmse, z_rmse, '..\\output\\ABC\\MatchResult_ABC_LIPoint.txt')

    elif name == 'PCB':

        # 导入原始数据
        lidar0 = np.loadtxt('..\\data\\PCB\\CSU_PCB_cut_02.txt', delimiter=',')
        lidar = lidar0[:, :3]
        dlg0 = np.loadtxt('..\\data\\PCB\\Polyline_PCB.txt', delimiter=',')
        dlg = dlg0[:, :3]

        # DoPP
        lidar_dopp = DoPP.DoPP.run(lidar, 1, 0.5, 10, name)

        # 图匹配
        R1, T1, lidar_dopp = GM_LIPoint.all_Ransac.run(dlg, lidar_dopp, name)
        print('图匹配所求参数')
        print(R1)
        print(T1)

        # ICP
        R2, T2, rmse = ICP.ICP(lidar_dopp, dlg[:, :2], 45, 100)
        print('ICP所求参数')
        print(R2)
        print(T2)

        # 刚体变换
        lidar[:, :2] = ICP.Transform(lidar[:, :2], R1, T1)
        lidar[:, :2] = ICP.Transform(lidar[:, :2], R2, T2)
        np.savetxt('..\\output\\PCB\\CSU_PCB_2D_LIPoint.txt', lidar, delimiter=',')

        # 高程配准
        VertailControlPoints = np.loadtxt('..\\data\\PCB\\CSU_PCB_VertailControlPoints.txt', delimiter=',')
        lidar, Z, z_rmse = Reg_H.registration_vertialPoints(VertailControlPoints, lidar, r=0.5)
        np.savetxt('..\\output\\PCB\\CSU_PCB_3D_LIPoint.txt', lidar, delimiter=',')
        GM_LIPoint.output(R1, T1, R2, T2, Z, rmse, z_rmse, '..\\output\\PCB\\MatchResult_PCB_LIPoint.txt')
