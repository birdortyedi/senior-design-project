from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import sklearn.naive_bayes as nb
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def knn_classifier(x_train, y_train, x_test):
    neigh = KNeighborsClassifier(n_neighbors=3, weights='distance', algorithm='ball_tree')
    neigh.fit(x_train, y_train)
    labels = neigh.predict(x_test)
    return labels

def svm_classifier(x_train, y_train, x_test):
    clf = svm.SVC(kernel='poly', degree=4, C=1, gamma=0.001)
    clf.fit(x_train, y_train)
    labels = clf.predict(x_test)
    return labels

def naive_bayes_classifier(x_train, y_train, x_test):
    gnb = nb.GaussianNB()
    gnb.fit(x_train, y_train)
    labels = gnb.predict(x_test)
    return labels


def decision_tree_classifier(x_train, y_train, x_test):
    dtc = tree.DecisionTreeClassifier(max_features=0.5, presort=True)
    dtc.fit(x_train, y_train)
    labels = dtc.predict(x_test)
    return labels

def random_forest_classifier(x_train, y_train, x_test):
    rfc = RandomForestClassifier(random_state=42, max_features=0.5, min_samples_leaf=2, n_estimators=999)
    rfc.fit(x_train, y_train)
    labels = rfc.predict(x_test)
    return labels

def evaluate(labels, y_test):
    correct = 0
    for i in range(len(y_test)):
        if labels[i] == y_test[i]:
            correct += 1
    accuracy = correct / len(y_test)
    print("Accuracy: %.3f" % accuracy)
