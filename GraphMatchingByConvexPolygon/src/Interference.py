# File      :Interference.py
# Author    :WJ
# Function  :
# Time      :2021/01/15
# Version   :
# Amend     :

import numpy as np
import math


def Offset(Pset,a=1):
    Pset1=np.array(Pset)
    for i in range(len(Pset1[0,:])):
        off = np.random.randn(len(Pset1))*a
        Pset1[:,i]=Pset1[:,i]+off
    return Pset1

def Rotation(Pset,angle=0):#绕质心旋转
    angle=math.radians(angle)
    mean=np.mean(Pset,axis=0)
    Pset1=Pset-mean
    R=np.array([[math.cos(angle),-math.sin(angle)],
                 [math.sin(angle),math.cos(angle)]])
    Pset2=np.dot(Pset1,R)
    Pset3=Pset2+mean
    return Pset3

def Shuffle(Pset):
    Pset=np.array(Pset)
    np.random.shuffle(Pset)
    return Pset

def Translation(Pset,x=0,y=0):
    Pset = np.array(Pset)
    X=np.ones(len(Pset))*x
    Y=np.ones(len(Pset))*y
    Pset[:,0]=X+Pset[:,0]
    Pset[:,1]=Y+Pset[:,1]
    return Pset

def Deletion(Pset,n=10):
    if len(Pset)-n<2:
        print('删除n个点后，数组内不到2个点，请重新确定将删除的点数。')
        return Pset
    else:
        Pset1=np.array(Pset)
        Pset1=Shuffle(Pset1)
        for i in range(0,n):
            Pset1=np.delete(Pset1,len(Pset1)-1,0)
        return Pset1




if __name__=='__main__':
    Pset0=np.array([[0.,1.],
                    [1.,1.],
                    [1.,0.],
                    [0.,0.]])

    Pset1=Translation(Pset0,1,1)
    print('--------------')
    print(Pset1)

    Pset2=Shuffle(Pset0)
    print('--------------')
    print(Pset2)

    Pset3=Rotation(Pset0,90)
    print('--------------')
    print(Pset3)

    Pset4=Offset(Pset0,1)
    print('--------------')
    print(Pset4)

    Pset5=Deletion(Pset0,2)
    print('--------------')
    print(Pset5)


