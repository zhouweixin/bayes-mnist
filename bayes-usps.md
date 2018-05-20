

```python
import scipy.io as scio
import numpy as np
from functools import reduce
import operator
```

# 读数据集函数


```python
"""
xfile: x文件
yfile: y文件
test_rate: 测试集比例
"""
def readData(xfile, yfile, test_rate):
    # 加载文件， 此处是dict
    x = scio.loadmat(xfile)
    y = scio.loadmat(yfile)
    
    x = x.get(xfile[8:-4])
    y = y.get(yfile[8:-4])
    
    # 变成从0-9
    y = y - 1
    
    # 二值化
    x = np.where(x > 0, 1, 0)
    
    num = x.shape[0]
    index = int(num * (1-test_rate))
    
    x_train = x[0:index]
    y_train = y[0:index]
    x_test = x[index:num]
    y_test = y[index:num]

    return x_train, y_train, x_test, y_test
```

# 读取训练集和测试集


```python
x_train, y_train, x_test, y_test = readData("dataset/usps_train.mat", "dataset/usps_train_labels.mat", 1/12)
print("x_train.shape = {}".format(x_train.shape))
print("y_train.shape = {}".format(y_train.shape))
print("x_test.shape = {}".format(x_test.shape))
print("y_test.shape = {}".format(y_test.shape))
```

    x_train.shape = (4261, 256)
    y_train.shape = (4261, 1)
    x_test.shape = (388, 256)
    y_test.shape = (388, 1)
    

# 计算流程p(y|x) = p(y)*p(x|y)
1. 计算p(y)
2. 计算p(x|y)
3. 计算p(y|x)

# 1.计算p(y)


```python
# 按y把数据分成10组
# 初始化px_group
px_group = []
for i in range(10):
    px_group.append(i)
    px_group[i] = []

# px_group的第一维表示类别[0-9], 第二维表示相应的所有x样例
for i, y in enumerate(y_train):
    px_group[int(y)].append(x_train[i])

py = []
for i in range(10):
    py.append(len(px_group[i]) / x_train.shape[0])

py = np.array(py)
py = py.reshape(-1, 1)
print("py.shape = {}".format(py.shape))
print("py       = {}".format(py.squeeze()))
```

    py.shape = (10, 1)
    py       = [ 0.16897442  0.13353673  0.10114996  0.08777282  0.08777282  0.07674255
      0.08847688  0.08425252  0.08167097  0.08965032]
    

# 2.计算p(x|y)


```python
pxy = []
for i in range(10):
    group = np.array(px_group[i])
    # 按列求和
    group = np.sum(group, axis=0) / len(px_group[i])
    pxy.append(group)

pxy = np.array(pxy)
print("pxy.shape = {}".format(pxy.shape))
```

    pxy.shape = (10, 256)
    

# 3.计算p(yi|x)

## (1) 多项式朴素贝叶斯


```python
def predict_mul(x):    
    result = []    
    for i in range(10):
        # 把0换成1
        temp = x * pxy[i]
        resulti = []
        for j in np.where(temp == 0, 1, temp):
            py_pxy = py[i][0] * reduce(operator.mul, j)
            resulti.append(py_pxy)
        result.append(resulti)
    
    result = np.argmax(result, axis=0)
    return result    
```

## (2) 伯努力朴素贝叶斯


```python
def predict_ber(x):    
    result = []    
    for i in range(10):
        # 把0换成1
        temp = x * pxy[i]
        resulti = []
        for j in np.where(temp == 0, 1-pxy[i], temp):
            py_pxy = py[i][0] * reduce(operator.mul, j)
            resulti.append(py_pxy)
        result.append(resulti)
    
    result = np.argmax(result, axis=0)
    return result  
```


```python
# 多项式朴素贝叶斯
train_result_mul = predict_mul(x_train)
test_result_mul = predict_mul(x_test)
# 伯努力朴素贝叶斯
train_result_ber = predict_ber(x_train)
test_result_ber = predict_ber(x_test)
```


```python
acc_train_mul = np.where(train_result_mul == np.squeeze(y_train), 1, 0)
print("训练集-多项式朴素贝叶斯 Acc_train_mul = {}".format(np.sum(acc_train_mul) / len(train_result_mul)))

acc_test_mul = np.where(test_result_mul == np.squeeze(y_test), 1, 0)
print("测试集-多项式朴素贝叶斯 Acc_test_mul = {}".format(np.sum(acc_test_mul) / len(test_result_mul)))

print()

acc_train_mul = np.where(train_result_ber == np.squeeze(y_train), 1, 0)
print("训练集-伯努力朴素贝叶斯 Acc_train_ber = {}".format(np.sum(acc_train_mul) / len(train_result_ber)))

acc_test_ber = np.where(test_result_ber == np.squeeze(y_test), 1, 0)
print("测试集-伯努力朴素贝叶斯 Acc_test_ber = {}".format(np.sum(acc_test_ber) / len(test_result_ber)))
```

    训练集-多项式朴素贝叶斯 Acc_train_mul = 0.7535789720722835
    测试集-多项式朴素贝叶斯 Acc_test_mul = 0.7551546391752577
    
    训练集-伯努力朴素贝叶斯 Acc_train_ber = 0.8385355550340295
    测试集-伯努力朴素贝叶斯 Acc_test_ber = 0.8247422680412371
    
