from json import load
from sklearn.datasets import load_boston
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


boston = load_boston()

X = boston.data
y = boston.target

l_reg = linear_model.LinearRegression()
plt.scatter(X.T[5],y)
plt.show()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = l_reg.fit(X_train, y_train)
predictions = model.predict(X_test)

print('prediction: ', predictions)
print('R^2 value: ', l_reg.score(X, y))
print('coedd: ', l_reg.coef_)         #.coef_是斜率
print('intercept: ', l_reg.intercept_)#.intercept_是截距 
