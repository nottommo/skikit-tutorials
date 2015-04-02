
from sklearn import datasets
from sklearn import svm
import pickle

iris = datasets.load_iris()
digits = datasets.load_digits()

# Create and clear a svm (support vector classification)
clf = svm.SVC(gamma=0.001, C=100.)
clf.fit(digits.data[:-1], digits.target[:-1])

# Dump and reload model
s = pickle.dumps(clf)
clf2 = pickle.loads(s)

# Predict the output
print "Predicted %d\nActual %d" % (clf2.predict(digits.data[-1]), digits.target[-1])
