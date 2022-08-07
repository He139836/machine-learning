import numpy as np
import operator
import matplotlib.pyplot as plt
import pandas
#加载数据集
df1 = pandas.read_csv('Breast_cancer_data.csv')
data1 = np.array(df1)
x_train=data1[:,0:5]
y_train=data1[:,5]
x_test=data1[:,0:5]
y_test=data1[:,5]
#分类函数
def classify(input, dataSet, labels, k):

    dataSetSize = dataSet.shape[0]
    # numpy中的tile方法,用于对矩阵进行填充
    diffMat = np.tile(input, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistance = sqDiffMat.sum(axis=1)
    distance = sqDistance ** 0.5
    # argsort()方法进行直接排序
    sortDist = distance.argsort()
    classCount = {}
    for i in range(k):
        # 取出前k个元素的类别
        voteIlabel = labels[sortDist[i]]
        # dict.get(key,default=None),字典的get()方法,返回指定键的值,如果值不在字典中返回默认值。
        # 计算类别次数
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # 排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # 返回次数最多的类别,即所要分类的类别
    return sortedClassCount[0][0]

#初始化数据集
group=x_train
labels =y_train
print(x_test.shape[0])
b=[]
# kNN分类
for i in range(569):
    y_pre = classify(x_test[i,0:5], group, labels, 3)
    b.append(y_pre)
# 打印分类结果
print(b)
add=y_test+b
sub=b-y_test

TP=np.sum(np.where(add==2,1,0))#正确正例个数
TN=np.sum(np.where(add==0,1,0))#正确反例个数
FP=np.sum(np.where(sub==1,1,0))#错误正例个数
FN=np.sum(np.where(sub==-1,1,0))#错误反例个数
print('TP',TP)
print('TN',TN)
print('FP',FP)
print('FN',FN)
#模型评估
accuracy=(TP+TN)/(TP+TN+FP+FN)#准确率
precision=TP/(TP+FP)#精确率
recall=TP/(TP+FN)#召回率
print('accuracy=',accuracy)
print('precision=',precision)
print('recall=',recall)




