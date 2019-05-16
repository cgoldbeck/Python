import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

#  Still don't know how classifier works...
features_origin = np.loadtxt('CollML_testdataset_features.dat')
labels_origin = np.loadtxt('CollML_testdataset_labels.dat')

#  V important, set test data
features, features_test, labels, labels_test = train_test_split(features_origin, labels_origin, test_size=0.2,
                                                                shuffle=False)
# Most of the time we want to shuffle but for leaning purposes we don't

plt.plot(features[labels == 0, 0], features[labels == 0, 1], 'bo')
plt.plot(features[labels == 1, 0], features[labels == 1, 1], 'rs')

plt.show()

#  Now lets classify
clf = DecisionTreeClassifier()

clf.fit(features, labels)

print(clf.score(features, labels))  # Overfitting....

export_graphviz(clf, 'Graph_DecisionTree.dat')  # Big ass tree

print(clf.feature_importances_) # Rely mostly on second feature
#  Underscore means object is created AFTER the fact i.e. after clf.fit is run

P1 = np.array([[-1, -1]])
P2 = np.array([[-1, 4]])

print(clf.predict(P1), clf.predict_proba(P1))
print(clf.predict(P2), clf.predict_proba(P2))
#  Over fitting man

#  Lets make a mesh grid
delta = 0.01
x = np.arange(-2.0, 5.001, delta)
y = np.arange(-2.0, 5.001, delta)

X, Y = np.meshgrid(x, y)

Z = clf.predict(np.c_[X.ravel(), Y.ravel()])
Z = Z.reshape(X.shape)

plt.contourf(X, Y, Z, cmap=plt.get_cmap('jet'))
#  plt.plot(features[labels == 0, 0], features[labels == 0, 1], 'bo')
#  plt.plot(features[labels == 1, 0], features[labels == 1, 1], 'rs')
plt.show()


#  Now lets classify... with depth
clf = DecisionTreeClassifier(max_depth=5)

clf.fit(features, labels)

print(clf.score(features, labels))  # Overfitting....
export_graphviz(clf, 'Graph_DecisionTree2.dat')  # Big ass tree

#  Lets make a mesh grid
delta = 0.01
x = np.arange(-2.0, 5.001, delta)
y = np.arange(-2.0, 5.001, delta)

X, Y = np.meshgrid(x, y)

Z = clf.predict(np.c_[X.ravel(), Y.ravel()])
Z = Z.reshape(X.shape)

plt.contourf(X, Y, Z, cmap=plt.get_cmap('jet'))
plt.plot(features[labels == 0, 0], features[labels == 0, 1], 'bo')
plt.plot(features[labels == 1, 0], features[labels == 1, 1], 'rs')
plt.show()

#  Random Forests
clf = RandomForestClassifier(n_estimators=50, max_depth=5, oob_score=True)
clf.fit(features, labels)
print(clf.oob_score_, clf.score(features, labels), clf.feature_importances_)

#  Lets make a mesh grid
delta = 0.01
x = np.arange(-2.0, 5.001, delta)
y = np.arange(-2.0, 5.001, delta)

X, Y = np.meshgrid(x, y)

Z = clf.predict(np.c_[X.ravel(), Y.ravel()])
Z = Z.reshape(X.shape)

plt.contourf(X, Y, Z, cmap=plt.get_cmap('jet'))
plt.xlim(-2, 5)
plt.ylabel(-2, 5)
#  plt.plot(features[labels == 0, 0], features[labels == 0, 1], 'bo')
#  plt.plot(features[labels == 1, 0], features[labels == 1, 1], 'rs')
plt.show()

#  Support Vector Machines
clf = SVC()
clf.fit(features, labels)
print(clf.score(features, labels))

clf = SVC(gamma=1000)
clf.fit(features, labels)
print(clf.score(features, labels))


def plotContours(clf, delta  =0.01):
    x = np.arange(-2.0, 5.001, delta)
    y = np.arange(-2.0, 5.001, delta)

    X, Y = np.meshgrid(x, y)

    Z = clf.predict(np.c_[X.ravel(), Y.ravel()])
    Z = Z.reshape(X.shape)

    plt.contourf(X, Y, Z, cmap=plt.get_cmap('jet'))
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

plotContours(clf)

clf = SVC(C=0.1, gamma=1)
clf.fit(features, labels)
print(clf.score(features, labels))
plotContours(clf)

#  Validation and Training

X_train, X_valid, Y_train, Y_valid = train_test_split(features, labels, test_size=0.33, shuffle=False)

print(X_train.shape, X_valid.shape)

setGammas = [0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300]
accuracies = []
for gamma in setGammas:
    clf = SVC(C=1, gamma=gamma)
    clf.fit(X_train, Y_train)
    accuracies.append(clf.score(X_valid, Y_valid))

plt.plot(setGammas, accuracies)
plt.ylabel('Accuracy')
plt.xlabel(r'$\gamma$')
plt.xscale('log')

plt.show()

#  Cross validation
kf = KFold(n_splits=4)

for train_index, valid_index in kf.split(features):
    X_train = features[train_index]
    Y_train = labels[train_index]
    X_valid = features[valid_index]
    Y_valid = labels[valid_index]

    clf = SVC(C=1., gamma=0.5)
    clf.fit(X_train, Y_train)
    print(clf.score(X_valid, Y_valid))

#  Now lets see how our classifiers perform

clf = RandomForestClassifier(n_estimators=50)
clf.fit(features, labels)
print("Random Forest Classifier")
print("Accuracy: ", accuracy_score(labels_test, clf.predict(features_test)))
print("Precision: ", precision_score(labels_test, clf.predict(features_test)))
print("Recall: ", recall_score(labels_test, clf.predict(features_test)))
print("F1-Score: ", f1_score(labels_test, clf.predict(features_test)))

clf = SVC(C=1, gamma=1.)
clf.fit(features, labels)
print("SVC")
print("Accuracy: ", accuracy_score(labels_test, clf.predict(features_test)))
print("Precision: ", precision_score(labels_test, clf.predict(features_test)))
print("Recall: ", recall_score(labels_test, clf.predict(features_test)))
print("F1-Score: ", f1_score(labels_test, clf.predict(features_test)))