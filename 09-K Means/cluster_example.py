# -*- coding:utf-8 -*-
"""
@author:Lisa
@file:cluster_example.py
@note:利用k-means实现地理坐标的聚类
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from bi_k_means import *

"""
函数：球面距离计算
原理：利用球面余弦定理计算球面上两点的距离
"""
def distSLC(vecA,vecB):
    a=np.sin(vecA[0,1]*np.pi/180) *np.sin(vecB[0,1]*np.pi/180)
    b=np.cos(vecA[0,1]*np.pi/180) *np.cos(vecB[0, 1] * np.pi /180)*\
      np.cos(np.pi*vecB[0,0]-vecA[0,0])/180

    return np.arccos(a+b)*6371.0

"""
函数：从文件中读取数据列表
注意：要根据文本格式来读取
"""
def loadSet(fileName):
    dataList=[]
    fr=open(fileName)
    for line in fr.readlines():
        curLine=line.split('\t')
        dataList.append([ float(curLine[4]),float(curLine[3])]) #第4和5列分别代表纬度和经度
    return np.mat(dataList)

"""
函数：绘制聚类点分布
"""
def draw(dataMat,centroids,clustAssign,k):
    fig = plt.figure()
    rect = [0.1, 0.1, 0.8, 0.8]  # 创建矩形
    # 创建不同标记图案
    scatterMarkers = ['s', 'o', '^', '8', 'p', \
                      'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0 = fig.add_axes(rect, label='ax0', **axprops)
    imgP = plt.imread('Portland.png')  # 导入地图
    ax0.imshow(imgP)
    ax1 = fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(k):
        ptsInCurrCluster = dataMat[np.nonzero(clustAssign[:, 0].A == i)[0], :]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:, 0].flatten().A[0], ptsInCurrCluster[:, 1].flatten().A[0], marker=markerStyle,
                    s=90)
    ax1.scatter(centroids[:, 0].flatten().A[0], centroids[:, 1].flatten().A[0], marker='+', s=300)
    plt.show()

if __name__=='__main__':
    dataMat=loadSet('places.txt')
    k=3
    centroids,clusterAssign=biKmeans(dataMat,k)
    draw(dataMat,centroids,clusterAssign,k)