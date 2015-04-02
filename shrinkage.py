#from __future__ import print_function
import numpy as np
from sklearn import linear_model, datasets

diabetes = datasets.load_diabetes()
diabetes_X_train = diabetes.data[:-20]
diabetes_X_test  = diabetes.data[-20:]
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test  = diabetes.target[-20:]

regr = linear_model.LinearRegression()
regr.fit(diabetes_X_train, diabetes_y_train)

print(regr.coef_)

# The mean square error
np.mean((regr.predict(diabetes_X_test)-diabetes_y_test)**2)

# Explained variance score: 1 is perfect prediction
# and 0 means that there is no linear relationship
# between X and Y.
regr.score(diabetes_X_test, diabetes_y_test)

X = np.c_[ .5, 1].T
y = [.5, 1]
test = np.c_[ 0, 2].T
regr = linear_model.Ridge(alpha=0.1)

import pylab as pl
pl.figure()

np.random.seed(0)
for _ in range(6):
   this_X = .1*np.random.normal(size=(2, 1)) + X
   regr.fit(this_X, y)
   pl.plot(test, regr.predict(test))
   pl.scatter(this_X, y, s=3)

pl.show()

alphas = np.logspace(-4, -1, 6)
print([regr.set_params(alpha=alpha
            ).fit(diabetes_X_train, diabetes_y_train,
            ).score(diabetes_X_test, diabetes_y_test) for alpha in alphas])
