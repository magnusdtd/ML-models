'''Linear_model trong sklearn không vẽ được parabola.
Vì thế trong các bài toán không gian 3 chiều (dữ liệu đầu ra y phụ thuộc vào 2 đại lượng khác nhau x1, x2) ta chỉ có thể vẽ được mảng cong đi qua chính giữa các điểm mẫu. 
y = ax1 + bx2 + c'''


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import linear_model

A = np.array([[2, 5, 7, 9, 11, 16, 19, 23, 22, 29, 29, 35, 37, 40, 46]]).T
b = np.array([[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]]).T

plt.plot(A, b, 'ro')

lr = linear_model.LinearRegression()
#Fit (train the model)
lr.fit(A, b)

#Draw graph
x0 = np.array([[1, 46]]).T
y0 = x0*lr.coef_ + lr.intercept_

plt.plot(x0, y0)    
plt.xlabel("x coordinate")
plt.ylabel("y coordinate")
plt.show()