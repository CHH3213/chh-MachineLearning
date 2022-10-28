# -*-coding:utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题
"""
岭回归
矩阵X不是满秩矩阵，非满秩矩阵在求逆时会出现问题。为了解决这个问题，统计学家引入岭回归（ridge regression）的概念。
"""
def loadDataSet(fileName):
	"""
	函数说明:加载数据
	Parameters:
		fileName - 文件名
	Returns:
		xArr - x数据集
		yArr - y数据集
	"""
	numFeat = len(open(fileName).readline().split('\t')) - 1
	xArr = []; yArr = []
	fr = open(fileName)
	for line in fr.readlines():
		lineArr =[]
		curLine = line.strip().split('\t')
		for i in range(numFeat):
			lineArr.append(float(curLine[i]))
		xArr.append(lineArr)
		yArr.append(float(curLine[-1]))
	return xArr, yArr

def ridgeRegres(xMat, yMat, lam = 0.2):
	"""
	函数说明:岭回归
	Parameters:
		xMat - x数据集
		yMat - y数据集
		lam - 缩减系数
	Returns:
		ws - 回归系数
	"""
	xTx = xMat.T * xMat
	denom = xTx + np.eye(np.shape(xMat)[1]) * lam
	if np.linalg.det(denom) == 0.0:
		print("矩阵为奇异矩阵,不能求逆")
		return
	ws = denom.I * (xMat.T * yMat)
	return ws

def ridgeTest(xArr, yArr):
	"""
	函数说明:岭回归测试
	Parameters:
		xMat - x数据集
		yMat - y数据集
	Returns:
		wMat - 回归系数矩阵
	"""
	xMat = np.mat(xArr); yMat = np.mat(yArr).T
	#数据标准化
	yMean = np.mean(yMat, axis = 0)		#行与行操作，求均值
	yMat = yMat - yMean			#数据减去均值
	xMeans = np.mean(xMat, axis = 0)	#行与行操作，求均值
	xVar = np.var(xMat, axis = 0)		#行与行操作，求方差
	xMat = (xMat - xMeans) / xVar		#数据减去均值除以方差实现标准化
	numTestPts = 30				#30个不同的lambda测试
	wMat = np.zeros((numTestPts, np.shape(xMat)[1]))	#初始回归系数矩阵
	for i in range(numTestPts):					#改变lambda计算回归系数
		ws = ridgeRegres(xMat, yMat, np.exp(i - 10))	#lambda以e的指数变化，最初是一个非常小的数，
		wMat[i, :] = ws.T 	#计算回归系数矩阵
	return wMat

def plotwMat():
	"""
	函数说明:绘制岭回归系数矩阵
	Website:
		http://www.cuijiahua.com/
	Modify:
		2017-11-20
	"""
	abX, abY = loadDataSet('abalone.txt')
	redgeWeights = ridgeTest(abX, abY)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(redgeWeights)	
	ax_title_text = ax.set_title(u'log(lambada)与回归系数的关系')
	ax_xlabel_text = ax.set_xlabel(u'log(lambada)')
	ax_ylabel_text = ax.set_ylabel(u'回归系数')
	plt.setp(ax_title_text, size = 20, weight = 'bold', color = 'red')
	plt.setp(ax_xlabel_text, size = 10, weight = 'bold', color = 'black')
	plt.setp(ax_ylabel_text, size = 10, weight = 'bold', color = 'black')
	plt.show()


def regularize(xMat, yMat):
	"""
	函数说明:数据标准化
	Parameters:
		xMat - x数据集
		yMat - y数据集
	Returns:
		inxMat - 标准化后的x数据集
		inyMat - 标准化后的y数据集
	"""	
	inxMat = xMat.copy()			#数据拷贝
	inyMat = yMat.copy()
	yMean = np.mean(yMat, 0)			#行与行操作，求均值
	inyMat = yMat - yMean			#数据减去均值
	inMeans = np.mean(inxMat, 0)   		#行与行操作，求均值
	inVar = np.var(inxMat, 0)     		#行与行操作，求方差
	inxMat = (inxMat - inMeans) / inVar		#数据减去均值除以方差实现标准化
	return inxMat, inyMat

def rssError(yArr,yHatArr):
	"""
	函数说明:计算平方误差
	Parameters:
		yArr - 预测值
		yHatArr - 真实值
	Returns:
		

	"""
	return ((yArr-yHatArr)**2).sum()



if __name__ == '__main__':
	plotwMat()