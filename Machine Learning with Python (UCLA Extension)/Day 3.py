import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  # Always expects 2D array, i.e [[]] not []
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier

#  Regression
numSamples = 100

linearCoef = 0.5
Intercept = 2.2

X = np.random.rand(numSamples)*10.0
e = np.random.rand(numSamples) - 0.5

print("Min of X: ", X.min())
print("Max of X: ", X.max())
print("Avg. error: ", e.mean())

Y = linearCoef*X + Intercept + e

plt.plot(X, Y, 'o', color=(0.2, 0.6, 1.0))
plt.xlabel('Feature')
plt.ylabel('Target')

plt.show()

X = X.reshape((numSamples, 1))

X_train, X_test, Y_train, Y_test, = train_test_split(X, Y, test_size=0.33)

model = LinearRegression()
model.fit(X_train, Y_train)

x_array = np.linspace(0, 10, 100)
y_array = model.predict(x_array.reshape((100, 1)))

plt.plot(X, Y, 'o', color=(0.2, 0.6, 1.0))
plt.plot(x_array, y_array, 'r-', linewidth=3.)
plt.xlabel('Feature')
plt.ylabel('Target')

plt.show()

print(model.coef_, model.intercept_)

print(model.score(X_test, Y_test))  # R^2

#  Poly Regression
e = e.reshape((len(e), 1))
Y = linearCoef*X + 0.15*X**2 + Intercept + e

plt.plot(X, Y, 'o', color=(0.2, 0.6, 1.0))
plt.xlabel('Feature')
plt.ylabel('Target')

plt.show()

X_train, X_test, Y_train, Y_test, = train_test_split(X, Y, test_size=0.33)

features = np.zeros((len(X_train), 2))
features[:, 0] = X_train[:, 0]
features[:, 1] = X_train[:, 0]**2

model = LinearRegression()
model.fit(features, Y_train)

print(model.coef_, model.intercept_)

x_array = np.linspace(0, 10, 100)
y_array = x_array*model.coef_[0, 0] + x_array**2*model.coef_[0, 1] + model.intercept_

plt.plot(X, Y, 'o', color=(0.2, 0.6, 1.0))
plt.plot(x_array, y_array, 'r-', linewidth=3.)
plt.xlabel('Feature')
plt.ylabel('Target')

plt.show()

#  Real Data
data = np.loadtxt('Regression_Exercise_dataset.dat')
print(data.shape)

#  ?np.loadtxt - useful way to examine function

Y_origin = data[:, 0]
X_origin = data[:, 1]

X, X_test, Y, Y_test = train_test_split(X_origin, Y_origin, test_size=0.2, shuffle=False)

plt.plot(X, Y, 'o')
plt.show()

X = X.reshape(X.shape[0], 1)

model =LinearRegression()
model.fit(X, Y)

coefs = []
degree = []

model =LinearRegression()
model.fit(X, Y)

degree.append(1)
coefs.append(np.abs(model.coef_).mean())

x_array = np.linspace(0, 1, 100)
x_array = x_array.reshape((len(x_array), 1))
y_array = model.predict(x_array)

plt.plot(X, Y, 'bo')
plt.plot(x_array, y_array, 'r-')
plt.show()

#  Use np.c_ concatenates columns
X_poly = np.c_[X, X**2]

model =LinearRegression()
model.fit(X_poly, Y)

degree.append(2)
coefs.append(np.abs(model.coef_).mean())

x_array = np.linspace(0, 1, 100)
x_array_poly = np.c_[x_array, x_array**2]
y_array = model.predict(x_array_poly)

plt.plot(X, Y, 'bo')
plt.plot(x_array, y_array, 'r-')
plt.show()

def getPoly(myArray, degree):
    result = np.zeros((myArray.shape[0], degree))
    for j in range(degree):
        result[:, j] = myArray.ravel()**(j+1)
    return result

d = 5
X_poly = getPoly(X, degree=d)
x_array_poly = getPoly(x_array, degree=d)

model =LinearRegression()
model.fit(X_poly, Y)

degree.append(d)
coefs.append(np.abs(model.coef_).mean())

y_array = model.predict(x_array_poly)

plt.plot(X, Y, 'bo')
plt.plot(x_array, y_array, 'r-')
plt.show()

d = 19
X_poly = getPoly(X, degree=d)
x_array_poly = getPoly(x_array, degree=d)

model =LinearRegression()
model.fit(X_poly, Y)

degree.append(d)
coefs.append(np.abs(model.coef_).mean())

y_array = model.predict(x_array_poly)

plt.plot(X, Y, 'bo')
plt.plot(x_array, y_array, 'r-')
plt.show()

#  Regularization and Ridge Regression
model = Ridge(alpha=0.1)  # Higher alpha can make fit less good, too much penality
model.fit(X_poly, Y)

y_array = model.predict(x_array_poly)

plt.plot(X, Y, 'bo')
plt.plot(x_array, y_array, 'r-')
plt.show()

model.coef_

#  Unsupervised Learning (e.g. clusters)

iris_data = load_iris()
print(iris_data.data.shape)
print(iris_data.feature_names)

plt.plot(iris_data.data[:, 1], iris_data.data[:, 2], 'o')
plt.show()

kmean = KMeans(n_clusters=3)
kmean.fit(iris_data.data)

cluster = kmean.predict(iris_data.data)
print(cluster.shape)
print(cluster)

index0 = cluster == 0
index1 = cluster == 1
index2 = cluster == 2

plt.plot(iris_data.data[index0, 1], iris_data.data[index0, 2], 'o', color='r', markersize=4)
plt.plot(iris_data.data[index1, 1], iris_data.data[index1, 2], 'o', color='k', markersize=4)
plt.plot(iris_data.data[index2, 1], iris_data.data[index2, 2], 'o', color='b', markersize=4)

plt.xlabel('Sepal width (cm)')
plt.ylabel('Petal width (cm)')

plt.show()

#  PCA

X = np.random.random((200, 3))
X[:, 2] = X[:, 0]

plt.matshow(np.cov(X.T))
plt.show()

print(np.cov(X.T))

pca = PCA()
pca.fit(X)

print(pca.explained_variance_ratio_)
print(pca.components_)

X_transform = pca.transform(X)
print(np.cov(X_transform.T))

plt.matshow(np.cov(X_transform.T))
plt.show()


#  Lets apply to breast cancer

bcancer = load_breast_cancer()

X, X_test, Y, Y_test = train_test_split(bcancer.data, bcancer.target, test_size=0.2, shuffle=False)

pca = PCA()
pca.fit(X)

plt.plot(pca.explained_variance_ratio_, 'o-')
plt.ylabel('Explained Variance')
plt.xlabel('PC Number')

X_PCAs = pca.transform(X)
X_PCAs = X_PCAs[:, :2]

index0 = Y == 0
index1 = Y == 1

plt.plot(X_PCAs[index0, 0], X_PCAs[index0, 1], 's', color='r')
plt.plot(X_PCAs[index1, 0], X_PCAs[index1, 1], 'o', color='b')

plt.show()

clf = RandomForestClassifier(n_estimators=100, max_depth=5)
clf.fit(X_PCAs, Y)

X_test_PCAs = pca.transform(X_test)
clf.score(X_test_PCAs[:, :2], Y_test)