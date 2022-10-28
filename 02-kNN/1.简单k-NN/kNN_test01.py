# -*- coding: UTF-8 -*-
import numpy as np

"""
函数说明:创建数据集

Parameters:
	无
Returns:
	group - 数据集
	labels - 分类标签

"""
def createDataSet():
	#四组二维特征
	group = np.array([[1,101],[5,89],[108,5],[115,8]])
	#四组特征的标签
	labels = ['爱情片','爱情片','动作片','动作片']
	return group, labels


def classify0(test_data, dataset, labels, k):
	"""
	函数说明:kNN算法,分类器

	Parameters:
		test_data - 用于分类的数据(测试集)
		dataSet - 用于训练的数据(训练集)
		labes - 分类标签
		k - kNN算法参数,选择距离最小的k个点
	Returns:
		sortedClassCount[0][0] - 分类结果


	"""
	# 计算距离
	dist = np.sum((test_data - dataset)**2, axis=1)**0.5
	# k个最近的标签。dist.argsort()表示返回排序后的数组数字在原数组中的索引
	k_labels = [labels[index] for index in dist.argsort()[0 : k]]
	# 出现次数最多的标签即为最终类别
	# 获取k_labels中出现次数最多的元素
	label= max(k_labels,key=k_labels.count)
	return label

if __name__ == '__main__':
	#创建数据集
	group, labels = createDataSet()
	#测试集
	test = [125,20]
	#kNN分类
	test_class = classify0(test, group, labels, 3)
	#打印分类结果
	print(test_class)
