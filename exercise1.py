from sklearn import datasets, neighbors, linear_model

# Load the data
digits = datasets.load_digits()
Xdigits = digits.data
Ydigits = digits.target

Xtrain = digits.data[:-10]
Ytrain = digits.target[:-10]
Xtest = digits.data[-10:]
Ytest = digits.target[-10:]

# Train the nearest neighbours model
knn = neighbors.KNeighborsClassifier()
knn.fit(Xtrain, Ytrain)
print knn.predict(Xtest)
print Ytest

# Train using a logistic regression model
reg = linear_model.LogisticRegression()
reg.fit(Xtrain, Ytrain)
print [int(round(i)) for i in reg.predict(Xtest)]
print Ytest
