import matplotlib
import matplotlib.pyplot as plt
import numpy as np 
from sklearn import linear_model
import matplotlib.animation as animation

fig, (ax1, ax2) = plt.subplots(1, 2)

#Data
A = np.array([[2, 9, 7, 9, 11, 16, 25, 23, 22, 29, 29, 35, 37, 40, 46]]).T
b = np.array([[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]]).T
ones = np.ones((A.shape[0], 1), dtype=np.int8)
A_ones = np.concatenate((A, ones), axis=1)
m = A_ones.shape[0]

'''----------------------------------------------------------'''

# cost(x) = 1/2m * (Ax-b)^2
def cost(x):
    return 0.5/m * np.linalg.norm(A_ones.dot(x) - b, 2)**2

#grad(x) = 1/m * A^T * (Ax-b)
def grad(x):
    return 1/m * A_ones.T.dot(A_ones.dot(x) - b)

def gradient_descent(x_init, learning_rate, iteration):
    x_list = [x_init]
    
    for i in range(iteration):
        x_new = x_list[-1] - learning_rate * grad(x_list[-1])

        if (np.linalg.norm(grad(x_new))/m < 1e-2): break

        x_list.append(x_new) 

    return x_list

def check_grad(x):
    eps = 1e-4
    g = np.zeros_like(x)
    
    for i in range(len(x)):
        x1 = x.copy()
        x2 = x.copy()
        x1[i] += eps
        x2[i] -= eps
        g[i] = (cost(x1) - cost(x2))/(2*eps)

    g_grad = grad(x)
    if (np.linalg.norm(g - g_grad) < 1e-5 ):
        print("No warning!")
    else:
        print("WARNING: CHECK GRADIENT FUNCTION!")


'''----------------------------------------------------------'''
    
fig.suptitle("GD for linear regression")
ax1.plot(A, b, 'ro')
ax1.set_xlim([0,60])
ax1.set_ylim([0,20])

#Line created by Linear Regression formula
lr = linear_model.LinearRegression()
lr.fit(A, b)
x0_gd = np.linspace(1, 46, 2)
y0_sklearn = lr.coef_[0][0]*x0_gd + lr.intercept_
ax1.plot(x0_gd, y0_sklearn, color="blue")

'''----------------------------------------------------------'''

#Random initial line
x_init = np.array([[1.], [2.]])
y0_init = x_init[0][0]*x0_gd + x_init[1][0]
ax1.plot(x_init, y0_init, color="black")

check_grad(x_init)

#Run gradient descent
iteration = 999
learning_rate = 0.0001
x_list = gradient_descent(x_init, learning_rate, iteration)
print(len(x_list))


#Draw x_list through gradient descent
for i in range(len(x_list)):
    y0_x_list = x_list[i][0][0]*x0_gd + x_list[i][1][0]
    ax1.plot(x0_gd, y0_x_list, color="green")

#Plot cost per iteration to determine when to stop
cost_list = []
iter_list = []
for i in range(len(x_list)):
    iter_list.append(i)
    cost_list.append(cost(x_list[i]))

ax2.plot(iter_list, cost_list)
ax2.set(xlabel="Iteration", ylabel="Cost value")



plt.show()