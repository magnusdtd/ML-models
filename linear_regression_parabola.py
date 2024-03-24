import numpy as np
import matplotlib
import matplotlib.pyplot as plt

#Random data
b = np.array([[2,5,7,9,11,16,19,23,22,29,29,35,37,40,46,42,39,31,30,28,20,15,10,6]]).T
A = np.array([[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]]).T

#Visualize data
plt.plot(A, b, 'ro')

#Create A square
x_square = np.array([A[:,0]**2]).T
A = np.concatenate((x_square, A), axis = 1)
# print(x_square)

#Create column vector with all of element is 1
ones = np.ones((A.shape[0], 1), dtype=np.int8)

#Conbine vector A and ones
A = np.concatenate((A, ones), axis=1)  
#axis = 1 in order to A not have only on column

#Fomular x = [A.(A^T)]^(-1)*(A^T)*b
x = np.linalg.inv(A.transpose().dot(A)).dot(A.transpose()).dot(b)
# y = ax^2+bx+c, x = [[a], [b], [c]]
# print(x)

#Test predicting data
x0 = np.linspace(1, 25, 10000)
y0 = x[0][0]*x0*x0 + x[1][0]*x0 + x[2][0]

plt.plot(x0, y0)    
plt.xlabel("x coordinate")
plt.ylabel("y coordinate")

x_test = 12
y_test = x_test*x_test*x[0][0] + x[1][0]*x_test + x[2][0]
print("x = 12 => y = "+str(y_test))

plt.show()