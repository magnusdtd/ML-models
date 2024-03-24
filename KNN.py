from sklearn import datasets
import numpy as np
import math
import operator

def get_distance(p1, p2):
    dimension = len(p1)
    distance = 0

    for i in range(dimension):
        distance += (p1[i] -  p2[i])*(p1[i] -  p2[i])

    return math.sqrt(distance)

def get_k_neighbors(training_x, label_y, point, k):
    distances = []
    neighbors = []

    for i in range(len(training_x)):
        distance = get_distance(point, training_x[i])
        distances.append((distance, label_y[i]))

    distances.sort(key=operator.itemgetter(0)) #sort by distance

    for i in range(k):
        neighbors.append(distances[i][1])

    return neighbors

def highest_votes(labels):
    labels_count = [0, 0, 0]
    for label in labels:
        labels_count[label] += 1

    max_count = max(labels_count)
    return labels_count.index(max_count)

def predict(training_x, label_y, point, k):
    neighbor_labels = get_k_neighbors(training_x, label_y, point, k)
    return highest_votes(neighbor_labels)

def accuracy_score(predicts, labels):
    total = len(predicts)
    correct_count = 0
    for i in range(total):
        if predicts[i] == labels[i]:
            correct_count += 1

    return correct_count/total #accuracy


def main():
    iris = datasets.load_iris()
    #data (petal length, petal width, sepal length, petal length)
    iris_x = iris.data 
    #label
    iris_y = iris.target 

    randIndex = np.arange(iris_x.shape[0])
    np.random.shuffle(randIndex)

    iris_x = iris_x[randIndex]
    iris_y = iris_y[randIndex]

    x_train = iris_x[:100, :]
    x_test = iris_x[100:, :]
    y_train = iris_y[:100]
    y_test = iris_y[100:]

    k = 10
    y_predict = []
    for p in x_test:
        label = predict(x_train, y_train, p, k)
        y_predict.append(label)


    print(y_test)
    print(y_predict)

    print(accuracy_score(y_test, y_predict))

main()
