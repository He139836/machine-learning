import numpy as np
import matplotlib.pyplot as plt
import pandas
#加载数据集
df1 = pandas.read_csv('Breast_cancer_data.csv')
data1 = np.array(df1)
x_train=data1[:,0:5]
y_train=data1[:,5]

x0=[]
x1=[]
#计算label_0的初始质心
for i in range(569):
    if y_train[i]==0:
        x0.append(x_train[i])
x0=np.array([x0])
x0=x0.sum(axis=1)
mean_0=x0/212
mean_0=mean_0.reshape(1,-1)
#计算label_1的初始质心
for i in range(569):
    if y_train[i]==1:
        x1.append(x_train[i])
x1=np.array([x1])
x1=x1.sum(axis=1)
mean_1=x1/357
mean_1=mean_1.reshape(1,-1)

costs_saved1=[]
costs_saved2=[]
#质心函数
def K_mean(x,y,center_0,center_1):
    for i in range(6) :
        for i in range(569):
            a=x[i]-center_0
            b=x[i]-center_1
            sqDiffMat1 = a**2
            sqDistance1 = sqDiffMat1.sum()**0.5
            sqDiffMat2 = b**2
            sqDistance2 = sqDiffMat2.sum()**0.5
            if sqDistance2 > sqDistance1:
                y[i]=0
            else:
                y[i]=1
        x0 = np.zeros((1,5))
        x1 = np.zeros((1,5))
        #两簇点分别求和
        for i in range(569):
            if y[i]==0:
                x0+=x[i]
            else:
                x1+=x[i]
        #均值算出新质心
        mean_0=x0/212
        mean_1=x1/357
        #误差计算，均方误差
        e1=(center_0-mean_0)
        e2=(center_1-mean_1)
        cost1=e1**2
        cost2=e2**2
        cost11=cost1.sum()
        cost22=cost2.sum()
        f1=cost11**0.5
        f2=cost22**0.5
        costs_saved1.append(f1.item(0))
        costs_saved2.append(f2.item(0))
        #赋值质心继续迭代
        center_0=mean_0
        center_1=mean_1

    return center_0,center_1

a,b=K_mean(x_train,y_train,mean_0,mean_1)
print("经过迭代最终的质心为")
print("label为0的质心：",a)
print("label为1的质心：",b)
plt.subplot(1,2,1)
plt.plot(range(1,np.size(costs_saved1)+1),costs_saved1,'r-o',linewidth=2,markersize=5)
plt.ylabel('Costs1')
plt.xlabel('iterations')

plt.subplot(1,2,2)
plt.plot(range(1,np.size(costs_saved2)+1),costs_saved2,'r-o',linewidth=2,markersize=5)
plt.ylabel('Costs2')
plt.xlabel('iterations')
plt.show()














