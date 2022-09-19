from sklearn import datasets                           #python內建的資料集
from sklearn.model_selection import train_test_split   #將資料切割成Training data(訓練資料）以及Test data(測試資料）

iris = datasets.load_iris()
X = iris.data
y = iris.target


# print(X.shape)
# print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2) #train_test_split(數據, random_state=亂樹種子, test_size = 拆分的比例)
print(X_train.shape)                    #.shape表示資料的維度
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)