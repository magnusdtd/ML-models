import numpy as np
import matplotlib
import matplotlib.pyplot as plt


#Random data
A = [2, 5, 7, 9, 11, 16, 19, 23, 22, 29, 29, 35, 37, 40, 46]
b = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

#Visualize data
plt.plot(A, b, 'ro')

#Change row vector to column vector
A = np.array([A]).T
b = np.array([b]).T

#Create column vector with all of element is 1
ones = np.ones((A.shape[0], 1), dtype=np.int8)

#Conbine vector A and ones
A = np.concatenate((A, ones), axis=1)  
#axis = 1 in order to A not have only on column

#Fomular x = [A.(A^T)]^(-1)*(A^T)*b
x = np.linalg.inv(A.transpose().dot(A)).dot(A.transpose()).dot(b)
# y = ax+b, x = [[a], [b]]
# print(x)

#Test predicting data
x0 = np.linspace(1, 46, 10000)
y0 = x[0][0]*x0 + x[1][0]

plt.plot(x0, y0)    
plt.xlabel("x coordinate")
plt.ylabel("y coordinate")

x_test = 12
y_test = x_test*x[0][0] + x[1][0]
print("x = 12 => y = "+str(y_test))

plt.show()