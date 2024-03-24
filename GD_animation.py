import matplotlib
import matplotlib.pyplot as plt
import numpy as np 
from sklearn import linear_model
import matplotlib.animation as animation

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

def update(i):
    y0_gd = x_list[i][0][0]*x0_gd + x_list[i][1][0]
    line.set_data(x0_gd, y0_gd)
    return line ,


'''----------------------------------------------------------'''
    
fig =plt.figure("GD for linear regression")
ax = plt.axes(xlim=(0, 60), ylim=(0,20))
plt.plot(A, b, 'ro')


#Line created by Linear Regression formula
lr = linear_model.LinearRegression()
lr.fit(A, b)
x0_gd = np.linspace(1, 46, 2)
y0_sklearn = lr.coef_[0][0]*x0_gd + lr.intercept_
ax.plot(x0_gd, y0_sklearn, color="blue")

'''----------------------------------------------------------'''

#Random initial line
x_init = np.array([[1.], [2.]])
y0_init = x_init[0][0]*x0_gd + x_init[1][0]
ax.plot(x_init, y0_init, color="black")

check_grad(x_init)

#Run gradient descent
iteration = 999
learning_rate = 0.0001
x_list = gradient_descent(x_init, learning_rate, iteration)
print(len(x_list))

# Draw x_list through gradient descent
for i in range(len(x_list)):
    y0_x_list = x_list[i][0][0]*x0_gd + x_list[i][1][0]
    ax.plot(x0_gd, y0_x_list, color="green", alpha=0.3)

#Draw animation
line , =  ax.plot([], [], color="purple")
iters = np.arange(1, len(x_list), 1)
line_animation = animation.FuncAnimation(fig, update, iters, interval=50, blit=True)

#Legend for the plot
plt.legend(("Value in each GD iteration", "Solution by formula ", "Initial value for GD"), loc=(0.52, 0.01))
ltext = plt.gca().get_legend().get_texts()

#Title
plt.title("Gadient Gradient Animation")

# plt.xlabel('Iteration: {}/{}, learning rate: {}'.format(2, len(x_list), learning_rate))

plt.show()