# File      :CornerPoints.py
# Author    :WJ
# Function  :
# Time      :2021/02/02
# Version   :
# Amend     :
import numpy as np
import cv2
from matplotlib import pyplot as plt
img = cv2.imread('dopp_result__c1_z5_t10.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
corners = cv2.goodFeaturesToTrack(gray,40,0.1,1)
# 返回的结果是[[ 311., 250.]] 两层括号的数组。
corners = np.int0(corners)
plt.figure(figsize=(16, 9))
for i in corners:
    x,y = i.ravel()
    cv2.circle(img,(x,y),3,255,-1)
plt.imshow(img),plt.show()
cv2.waitKey()
