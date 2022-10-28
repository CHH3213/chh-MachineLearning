import numpy as np
"""KNN 分类器实现
k-近邻算法的一般流程：

    计算已知类别数据集中的点与当前点之间的距离；
    按照距离递增次序排序；
    选取与当前点距离最小的k个点；
    确定前k个点所在类别的出现频率；
    返回前k个点所出现频率最高的类别作为当前点的预测分类。
"""
def classify0(test_data,dataSet,labels,k):
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
    dist = []
    for data in dataSet:
        di = np.sqrt((test_data[0]-data[0])**2+(test_data[1]-data[1])**2)
        dist.append(di)
    dist = np.array(dist)
    print(dist)
    k_labels = [labels[ind] for ind in dist.argsort()[0:k]]
    label = max(k_labels,key=k_labels.count)
    return label
    

