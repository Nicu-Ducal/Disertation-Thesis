import numpy as np
import matplotlib.pyplot as plt
from BaseSVDD import BaseSVDD, BananaDataset

# Banana-shaped dataset generation and partitioning
for i in range(3):
    # X, y = BananaDataset.generate(number=100, display='on')
    n = 1000
    dim = 2
    X = np.r_[np.random.randn(n, dim)]

    y = None
    # X_train, X_test, y_train, y_test = BananaDataset.split(X, y, ratio=0.3)

    print(X.shape)
    # 
    svdd = BaseSVDD(C=0.9, gamma=0.3, kernel='rbf', display='on')

    # # # 
    svdd.fit(X,  y)

    print(svdd.radius, svdd.offset, svdd.get_distance(X[:1]), X[:1], svdd.predict(X[:1]))

    # testing
    dists = svdd.get_distance(X)
    print(dists.shape)
    radii = np.linspace(0, 1.5, 200)
    outliers = np.array([np.sum(svdd.get_distance(X) > r) for r in radii])

    print(outliers)

    # plt.plot(radii, outliers)
    plt.clf()

    plt.plot(radii, outliers)
    plt.plot(svdd.radius, np.sum(svdd.get_distance(X) > svdd.radius), 'ro')
    plt.xlabel('raza')
    plt.ylabel('nr. outliers')
    plt.title("Outliers in functie de raza")
    # plt.legend()
plt.show()

from sklearn.svm import OneClassSVM, LinearSVC, SVC

# # # 
# svdd.plot_boundary(X_train,  y_train)

# # #
# y_test_predict = svdd.predict(X_test, y_test)

# # #
# radius = svdd.radius
# distance = svdd.get_distance(X_test)
# svdd.plot_distance(radius, distance)
