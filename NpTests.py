import numpy as np
from sklearn.naive_bayes import GaussianNB
import random
import pylab as pl
import matplotlib

matplotlib.use('agg')

import matplotlib.pyplot as plt

x_min = 0.1; x_max = 1
y_min = 0.1; y_max = 1
h = .25
x_arranged = np.arange(x_min, x_max, h)
y_arranged = np.arange(y_min, y_max, h)
xx, yy = np.meshgrid(x_arranged, y_arranged)
print xx
print yy
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

n_points = 10
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

clf = GaussianNB()
clf.fit(X_train, y_train)

print xx.ravel()
print yy.ravel()
ravel_ = np.c_[xx.ravel(), yy.ravel()]
print ravel_
predicted = clf.predict(ravel_)
print predicted
reshaped = predicted.reshape(xx.shape)
print reshaped
plt.pcolormesh(xx, yy, reshaped)

plt.savefig("nptest.png")



