from sklearn import cross_validation, datasets, linear_model
import numpy as np

diabetes = datasets.load_diabetes()
X = diabetes.data[:150]
y = diabetes.target[:150]
xtest = diabetes.data[150:]
ytest = diabetes.target[150:]

alphas = np.logspace(-4, -.5, 30)
lasso = linear_model.LassoCV(alphas=alphas)
lasso.fit(X,y)

lasso

diff = lasso.predict(xtest) - ytest
print np.mean(ytest)
print np.mean(diff)
print np.std(diff)
