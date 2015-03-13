import random
import pylab as pl
import numpy as np
import matplotlib

matplotlib.use('agg')

import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

def prettyPicture(clf, X_test, y_test, file_name):
    x_min = 0.0; x_max = 1.0
    y_min = 0.0; y_max = 1.0
    h = .01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.pcolormesh(xx, yy, Z, cmap=pl.cm.seismic)
    grade_sig = [X_test[ii][0] for ii in range(0, len(X_test)) if y_test[ii]==0]
    bumpy_sig = [X_test[ii][1] for ii in range(0, len(X_test)) if y_test[ii]==0]
    grade_bkg = [X_test[ii][0] for ii in range(0, len(X_test)) if y_test[ii]==1]
    bumpy_bkg = [X_test[ii][1] for ii in range(0, len(X_test)) if y_test[ii]==1]
    plt.scatter(grade_sig, bumpy_sig, color = "b", label="fast")
    plt.scatter(grade_bkg, bumpy_bkg, color = "r", label="slow")
    plt.legend()
    plt.xlabel("bumpiness")
    plt.ylabel("grade")
    plt.savefig(file_name)

def makeTerrainData(n_points=1000):
    random.seed(42)
    grade = [random.random() for ii in range(0,n_points)]
    bumpy = [random.random() for ii in range(0,n_points)]
    error = [random.random() for ii in range(0,n_points)]
    y = [round(grade[ii]*bumpy[ii]+0.3+0.1*error[ii]) for ii in range(0,n_points)]
    for ii in range(0, len(y)):
        if grade[ii]>0.8 or bumpy[ii]>0.8:
            y[ii] = 1.0

    X = [[gg, ss] for gg, ss in zip(grade, bumpy)]
    split = int(0.75*n_points)
    X_train = X[0:split]
    X_test  = X[split:]
    y_train = y[0:split]
    y_test  = y[split:]
    return X_train, y_train, X_test, y_test

def getAccuracy(features_test, labels_test, clf, name):
    accuracy = clf.score(features_test, labels_test)
    print name + ": " + str(accuracy)

def classifyBayes(features_train, labels_train, features_test, labels_test):
    clf = GaussianNB()
    clf.fit(features_train, labels_train)
    prettyPicture(clf, features_test, labels_test, "bayes.png")
    getAccuracy(features_test, labels_test, clf, "Bayes")

def classifySVM(features_train, labels_train, features_test, labels_test):
    clf = SVC(kernel="rbf", C=1000000.0)
    clf.fit(features_train, labels_train)
    prettyPicture(clf, features_test, labels_test, "svm.png")
    getAccuracy(features_test, labels_test, clf, "svm")

def classifyDT(features_train, labels_train, features_test, labels_test):
    clf = DecisionTreeClassifier(min_samples_split=50)
    clf.fit(features_train, labels_train)
    prettyPicture(clf, features_test, labels_test, "dt.png")
    getAccuracy(features_test, labels_test, clf, "dt")

def classifyKNearest(features_train, labels_train, features_test, labels_test):
    clf = KNeighborsClassifier(n_neighbors=20)
    clf.fit(features_train, labels_train)
    prettyPicture(clf, features_test, labels_test, "k-nearest.png")
    getAccuracy(features_test, labels_test, clf, "k-nearest")

def classifyRandomForest(features_train, labels_train, features_test, labels_test):
    clf = RandomForestClassifier(min_samples_split=50, n_estimators=20)
    clf.fit(features_train, labels_train)
    prettyPicture(clf, features_test, labels_test, "forest.png")
    getAccuracy(features_test, labels_test, clf, "forest")

def classifyAdaBoost(features_train, labels_train, features_test, labels_test):
    clf = AdaBoostClassifier(n_estimators=100)
    clf.fit(features_train, labels_train)
    prettyPicture(clf, features_test, labels_test, "adaboost.png")
    getAccuracy(features_test, labels_test, clf, "adaboost")

def main():
    features_train, labels_train, features_test, labels_test = makeTerrainData()

    classifyBayes(features_train, labels_train, features_test, labels_test)
    classifySVM(features_train, labels_train, features_test, labels_test)
    classifyDT(features_train, labels_train, features_test, labels_test)
    classifyKNearest(features_train, labels_train, features_test, labels_test)
    classifyRandomForest(features_train, labels_train, features_test, labels_test)
    classifyAdaBoost(features_train, labels_train, features_test, labels_test)

main()
