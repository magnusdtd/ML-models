from sklearn import datasets, neighbors
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt

def main():
    digit = datasets.load_digits()
    #data (petal length, petal width, sepal length, petal length)
    digit_x = digit.data 
    #label
    digit_y = digit.target 

    randIndex = np.arange(digit_x.shape[0])
    np.random.shuffle(randIndex)

    digit_x = digit_x[randIndex]
    digit_y = digit_y[randIndex]

    x_train, x_test, y_train, y_test = train_test_split(digit_x, digit_y, test_size=360)

    knn = neighbors.KNeighborsClassifier(n_neighbors=5)
    knn.fit(x_train, y_train)

    y_predict = knn.predict(x_test)

    print(accuracy_score(y_predict, y_test))
    x = int(input("Enter index of the image you want to test (1, 360): ")) - 1
    plt.gray()
    plt.imshow(x_test[x].reshape(8, 8))
    print(knn.predict(x_test[x].reshape(1, -1)))
    plt.show()

main()