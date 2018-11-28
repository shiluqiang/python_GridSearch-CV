# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 10:35:11 2018

@author: lj
"""
import numpy as np
from sklearn import svm
from sklearn import cross_validation
from sklearn.model_selection import GridSearchCV

def load_data(data_file):
    '''导入训练数据
    input:  data_file(string):训练数据所在文件
    output: data(mat):训练样本的特征
            label(mat):训练样本的标签
    '''
    data = []
    label = []
    f = open(data_file)
    for line in f.readlines():
        lines = line.strip().split(' ')
        
        # 提取得出label
        label.append(float(lines[0]))
        # 提取出特征，并将其放入到矩阵中
        index = 0
        tmp = []
        for i in range(1, len(lines)):
            li = lines[i].strip().split(":")
            if int(li[0]) - 1 == index:
                tmp.append(float(li[1]))
            else:
                while(int(li[0]) - 1 > index):
                    tmp.append(0)
                    index += 1
                tmp.append(float(li[1]))
            index += 1
        while len(tmp) < 13:
            tmp.append(0)
        data.append(tmp)
    f.close()
    return np.array(data), np.array(label).T

### 1.导入数据集
trainX,trainY = load_data('heart_scale')

### 2.设置C和sigma的取值范围
c_list = []
for i in range(1,50):
    c_list.append(i * 0.5)
    
gamma_list = []
for j in range(1,40):
    gamma_list.append(j * 0.2)
    
### 3.1循环嵌套实现网格搜索 + 交叉验证
best_value = 0.0

for i in c_list:
    for j in gamma_list:
        current_value = 0.0
        rbf_svm = svm.SVC(kernel = 'rbf', C = i, gamma = j)
        scores = cross_validation.cross_val_score(rbf_svm,trainX,trainY,cv =3,scoring = 'accuracy')
        current_value = scores.mean()
        if current_value >= best_value:
            best_value = current_value
            best_parameters = {'C': i, 'gamma': j}
        print('Best Value is :%f'%best_value)
        print('Best Parameters is',best_parameters)

### 3.2GridSearchCV(网格搜索+CV)
param_grid = {'C': c_list,
              'gamma':gamma_list}

rbf_svm1 = svm.SVC(kernel = 'rbf')
grid = GridSearchCV(rbf_svm1, param_grid, cv=3, scoring='accuracy')
grid.fit(trainX,trainY)
best_parameter = grid.best_params_
print(best_parameter)



            
        












