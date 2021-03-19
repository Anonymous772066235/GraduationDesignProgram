# File      :CompareHist.py
# Author    :WJ
# Function  :
# Time      :2021/01/11
# Version   :
# Amend     :



import numpy as np
from matplotlib import pyplot as plt




def hist(array=[0,1],bins=20,start=0,end=1):
    hist=np.zeros((bins+1,1))
    x=np.linspace(start=start,stop=end,num=bins+1,endpoint=True)
    for i in range(0,len(array)):
        for j in range(0,len(x)-1):
            if abs(array[i])>=x[j] and abs(array[i])<x[j+1]:
                hist[j]=hist[j]+1
    return hist





def calcSimilarity(hist1, hist2,Similarity=False):
    # 若Simlarity==Ture，则返回相似度，否则返回相异度
    # 计算两直方图的相似度
    # 计算直方图的重合度
    degree = 0
    for i in range(len(hist1)):
        if hist1[i] != hist2[i]:
            degree = degree + \
                (1 - abs(abs(hist1[i]) - abs(hist2[i]))/ max(abs(hist1[i]), abs(hist2[i])))
        else:
            degree = degree + 1
    degree = degree / len(hist1)
    if Similarity==True:
        return degree
    else:
        return 1-degree



def calaHistCorrelation(U1k,U2l,bins=10,start=0,end=1):
    hist1=hist(U1k,bins,start,end)
    hist2=hist(U2l,bins,start,end)
    CorrelationDegree = calcSimilarity(hist1, hist2)
    return CorrelationDegree


if __name__=='__main__':
    U = np.array([-0.1, -0.1, -0.2, -0.9, 0, 0.5, 0.2, 0.6, 0.63, 0.7, 0.8])
    hist1 = hist(U)
    hist2 =np.array(hist1)
    hist2[2]=3
    plt.subplot(1, 2, 1)
    plt.title("hist1")
    plt.plot(hist1)
    plt.subplot(1, 2, 2)
    plt.title("hist2")
    plt.plot(hist2)
    plt.pause(5)
    degree=calcSimilarity(hist1,hist2)
    print(degree)