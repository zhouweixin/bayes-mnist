

```python
import scipy.io as scio
import numpy as np
from functools import reduce
import operator
from sklearn.ensemble import RandomForestClassifier
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
    
    # 二值化
    x = np.where(x > 0, 1, 0)
    
    num = x.shape[0]
    index = int(num * (1-test_rate))
    
    x_train = x[0:index]
    y_train = y[0:index]
    x_test = x[index:num]
    y_test = y[index:num]
    
    y_train = y_train.ravel()
    y_test = y_test.ravel()

    return x_train, y_train, x_test, y_test
```

# 读取数据集


```python
x_train, y_train, x_test, y_test = readData("dataset/usps_train.mat", "dataset/usps_train_labels.mat", 1/12)
print("x_train.shape = {}".format(x_train.shape))
print("y_train.shape = {}".format(y_train.shape))
print("x_test.shape = {}".format(x_test.shape))
print("y_test.shape = {}".format(y_test.shape))
```

    x_train.shape = (4261, 256)
    y_train.shape = (4261,)
    x_test.shape = (388, 256)
    y_test.shape = (388,)
    

# 开始训练


```python
rfc = RandomForestClassifier(n_estimators=500, oob_score=True, n_jobs=4)
model = rfc.fit(x_train, y_train)
```

# 准确率


```python
print("训练集-带外准确率 Acc_train_usps= {}".format(rfc.oob_score_))
print("测试集-准确率 Acc_test_usps = {}".format(rfc.score(x_test, y_test)))
```

    训练集-带外准确率 Acc_train_usps= 0.9380427129781741
    测试集-准确率 Acc_test_usps = 0.9407216494845361
    
