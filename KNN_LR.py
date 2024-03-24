import numpy as np 
from matplotlib import pyplot as plt 
from sklearn import neighbors

def main ():
    X = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    Y = [2, 5, 7, 9, 11, 16, 19, 23, 22, 29, 29, 35, 37, 40, 46, 42, 39, 31, 30, 28, 20, 15, 10, 6]
    plt.plot(X, Y, 'ro')
    X = np.array(X).reshape(-1, 1)
    Y = np.array(Y)

    x0 = np.linspace(3, 25, 10000).reshape(-1, 1)
    y0 = []

    knn = neighbors.KNeighborsClassifier(n_neighbors=3)
    knn = knn.fit(X, Y)
    y0 = knn.predict(x0)

    test = [[15.]]
    print(knn.predict(test))

    plt.plot(x0, y0)
    plt.show()

main()