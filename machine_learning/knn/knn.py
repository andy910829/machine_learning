import numpy as np
import pandas as pd
from sklearn import neighbors, metrics   #提供了針對無監督和受監督的基於鄰居的學習方法的功能  https://iter01.com/549769.html
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder  #Label encoding : 把每個類別 mapping 到某個整數，不會增加新欄位 ;One hot encoding : 
#為每個類別新增一個欄位，用 0/1 表示是否  https://medium.com/@PatHuang/%E5%88%9D%E5%AD%B8python%E6%89%8B%E8%A8%98-3-%E8%B3%87%E6%96%99%E5%89%8D%E8%99%95%E7%90%86-label-encoding-one-hot-encoding-85c983d63f87

#數據來源 https://archive.ics.uci.edu/ml/datasets/car+evaluation
data = pd.read_csv('knn\car.data')
# print(data.head())


X = data[[
    'buying',
    'maint',
    'safty',
    'doors',
    'lug_boot'
]].values          #X,y皆為二維陣列
y = data[['class']]
#print(y)
# print(X,y)

#X
Le = LabelEncoder()
for i in range(len(X[0])):
    X[:, i] = Le.fit_transform(X[:, i]) #將所有字串轉換為數字代替，這組數據有大瞭之分所以可以用Label encoding，若無大小之分例如:國家名稱，可以用One hot encoding
#https://medium.com/@PatHuang/%E5%88%9D%E5%AD%B8python%E6%89%8B%E8%A8%98-3-%E8%B3%87%E6%96%99%E5%89%8D%E8%99%95%E7%90%86-label-encoding-one-hot-encoding-85c983d63f87
#X[:,0]是numpy中數組的一種寫法，表示對一個二維數組，取該二維數組第一維中的所有數據，
#第二維中取第0個數據，直觀來說，X[:,0]就是取所有行的第0個數據, X[:,1] 就是取所有行的第1個數據。
#https://blog.csdn.net/a394268045/article/details/79104219

#print(X)
#y
label_mapping = {
    'unacc':0,
    'acc':1,
    'good':2,
    'vgood':3
}
y['class'] = y['class'].map(label_mapping) #map()會根據提供的函數對指定順序做映射。 https://www.runoob.com/python/python-func-map.html
y = np.array(y)
#print(y)

#create model

knn = neighbors.KNeighborsClassifier(n_neighbors=25, weights='uniform')   #n_neighbors: 一個整數，指定k值; 
#weight：一個字符串或者可調用對象，指定投票權重類型。也就是這些鄰居投票可以相同或者不同（uniform：本節點的所有鄰居節點的投票權重都相等。
#distance:本節點的所有鄰居節點的投票權重和距離成反比。callable：一個可調用對象，他傳入距離的數組，返回同樣形狀的數組）。
#https://twgreatdaily.com/BGKX7W4BMH2_cNUgFg2t.html
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  #X_train是題目，y_train是解答，X_test是考試，y_test是考試的答案
knn.fit(X_train, y_train)
prediction = knn.predict(X_test)
accuracy = metrics.accuracy_score(y_test, prediction)
print('prediction: ', prediction)
print('accuracy: ', accuracy)

a = 100
print('actual value: ', y[a])
print('predicted value: ', knn.predict(X)[a])