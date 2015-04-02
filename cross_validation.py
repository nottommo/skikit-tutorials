import numpy as np
from sklearn import cross_validation, datasets, svm
import matplotlib.pyplot as plt

digits = datasets.load_digits()
X = digits.data
y = digits.target

C_s = np.logspace(-10, 0, 10)

cvs = []

for c in C_s:
    svc = svm.SVC(kernel='linear',C=c)
    cv = np.mean(cross_validation.cross_val_score(svc, X, y, n_jobs=-1))
    cvs.append(cv)

plt.semilogx(C_s, cvs)
plt.show()
