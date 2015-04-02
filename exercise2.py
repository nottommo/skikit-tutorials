from sklearn import datasets, neighbors, linear_model, svm, cross_validation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Import the iris data set and extract only the first two classes
iris = datasets.load_iris()
X = iris.data
y = iris.target
X = X[y != 0, :2]
y = y[y != 0]

h = .02  # step size in the mesh

# Generate a random permutation of this size
r = np.arange(len(y))
np.random.shuffle(r)

# Get the test and train datasets
xtrain = X[r[:-10]]
ytrain = y[r[:-10]]
xtest = X[r[-10:]]
ytest = y[r[-10:]]

# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

for i, model in enumerate(['linear', 'poly', 'rbf']):
    # Train using linear kernel
    svc = svm.SVC(kernel=model)
    svc.fit(xtrain, ytrain)
    ypred = svc.predict(xtest)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.subplot(1, 3, i)
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    plt.scatter(xtrain[:,0], xtrain[:,1], c=ytrain, cmap=cmap_bold)
    plt.scatter(xtest[:,0], xtest[:,1], c=ypred, s=100, cmap=cmap_bold)

plt.show()
