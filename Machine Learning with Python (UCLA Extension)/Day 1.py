#  They are doing most the work in Jupyter notebook, I will be using PyCharm

x = range(10)  # Starts at 0, ends at n-1
for element in x:
    print("Element", element)

#  Everything is an object
print(x.start, x.step, x.stop)

#  Beginning comments should have two spaces
print("Test")  # Inline comments should have two space before and one after

#  Some basic maths
x = 5*3 + 2
print(x)

y = 7*x**2 + 0.6*x
print(y)

z = 1 / x  # Unsure if correct spacing style guide
print(z)

print(y*x)

#  Some libraries, Numpy, Matplotlib, and Scikit
import numpy as np

a = [1, 2, 3, 4, 5]

b = np.array([1, 2, 3,4, 5])

print(type(a), type(b))

c = np.array([5, 4, 3, 2, 1])

print(a + a, b + c)  # concatenation vs pairwise addition

print(b + 7.1)
print(b*2.5)
print(b**3)
print(b + c)
print(b*c)
print(b**c)
print(b/c)
print(b % c)

np.exp(b)
np.cos(b)
np.sin(b)

print(b[0])
print(b[1])
print(b[-1])
print(b[0:-1:2])

print(b.shape)
b.sum()

np.arange(1, 5, .5)
np.linspace(0, 10, 4)  # n points meals n-1 spaces i.e. 4 points is 3 sections
np.ones(5)
np.zeros(5)

A = np.array([[1, 2, 3, 4], [10, 20, 30, 40], [100, 200, 300, 400]])  # Don't forget whitespace
print(A)
print(A.shape)
print(A[1, 2])
print(A[1, :])
print(A[:, 3])

#  1D array vs 2D array (vector)
B = np.arange(10)
print(B.shape)
print(B.ndim)

B = B.reshape((10, 1))
print(B.shape)
print(B.ndim)

print(A > 25)
print(A[A > 25])


import matplotlib.pyplot as plt

y = np.array([0, 10, 3, 4, 2])
plt.plot(y)
plt.show()

x = np.arange(15)
y1 = np.sin(0.5*x)
y2 = np.cos(0.5*x)

plt.plot(x, y1, 'o--', markersize=10, linewidth=1.2, color='r', label='Sine')
plt.plot(x, y2, 's--', markersize=10, linewidth=1.2, color='k', label='Cosine')

plt.xlabel("Time (s)")
plt.ylabel("Fluorescence (a.u)")

plt.yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
plt.legend(frameon=False)
plt.show()


#  Now lets do some machine learning
#  Mainly Supervised Learning (regression and classification)

#  Random forests not terrible

import sklearn

import sklearn.datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pandas as pd

bcancer = sklearn.datasets.load_breast_cancer()

print("Features: ", bcancer.data.shape)
print("Target: ", bcancer.target.shape)
print(type(bcancer.data))

print(bcancer.feature_names)

bcancer_model = DecisionTreeClassifier(max_depth=4)
bcancer_model.fit(bcancer.data, bcancer.target)

print(bcancer_model.predict([bcancer.data[0]]))
print(bcancer.target[0])


print(bcancer_model.predict([bcancer.data[19]]))
print(bcancer.target[19])

bcancer_predictions = bcancer_model.predict(bcancer.data)

compare = (bcancer.target == bcancer_predictions).astype(int)
plt.hist(compare, bins=3)
plt.ylabel('Count')
plt.xticks([0.15, 0.85], ['Misclassified', 'Correctly classified'])

plt.show()

print(compare.sum()/len(compare))

#  Oops, no testing data

bcancer.data_train, bcancer.data_test, bcancer.target_train, bcancer.target_test = train_test_split(bcancer.data,
                                                                                                    bcancer.target,
                                                                                                    test_size=0.2)

print("Train Size: ", bcancer.data_train.shape, bcancer.target_train.shape)
print("Test Size: ", bcancer.data_test.shape, bcancer.target_test.shape)

bcancer_model2 = DecisionTreeClassifier(max_depth=4)
bcancer_model2.fit(bcancer.data_train, bcancer.target_train)

bcancer_predictions2 = bcancer_model2.predict(bcancer.data_test)

compare2 = (bcancer.target_test == bcancer_predictions2).astype(int)
plt.hist(compare2, bins=3)
plt.ylabel('Count')
plt.xticks([0.15, 0.85], ['Misclassified', 'Correctly classified'])

plt.show()

print(compare2.sum()/len(compare2))
print(accuracy_score(bcancer.target_test, bcancer_predictions2))

X = pd.DataFrame(
    confusion_matrix(bcancer_predictions2, bcancer.target_test),
    index=['Predicted Not Cancer', 'Predicted Cancer'],
    columns=['True Not Cancer', 'True Cancer'])

XX = X/X.sum()
YY = pd.DataFrame.stack(XX)
YY.index = ['Specificity', 'False Negative', 'False Positive', 'Sensitivity']
