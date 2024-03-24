import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def draw_data(A, b, x0, y0):

    #Data
    fig =plt.figure("GD for linear regression")
    ax = plt.axes(xlim=(-10, 40), ylim=(-30,60))
    plt.plot(A, b, "ro")

    #Parabola with linear regression
    plt.plot(x0, y0)

    return fig, ax

def munipulate_A_data(A):
    x_square = np.array([A[:, 0]**2]).T
    A = np.concatenate((x_square, A), axis=1)

    ones = np.ones((A.shape[0], 1), dtype=np.int8)
    A = np.concatenate((A ,ones), axis=1)

    return A

def create_parabola(A_ones, b, x0):
    #Fomular x = [A.(A^T)]^(-1)*(A^T)*b
    x = np.linalg.inv(A_ones.transpose().dot(A_ones)).dot(A_ones.transpose()).dot(b)
    
    y0 = x[0][0]*x0*x0 + x[1][0]*x0 + x[2][0]
    return x0, y0

def cost(A_ones, b, x):
    # cost(x) = 1/2m * (Ax-b)^2
    return 0.5/A_ones.shape[0] * np.linalg.norm(A_ones.dot(x) - b, 2)**2

def grad(A_ones, b, x):
    #grad(x) = 1/m * A^T * (Ax-b)
    return 1/A_ones.shape[0] * A_ones.T.dot(A_ones.dot(x) - b)

def GD_create_parabolas(root_init, learning_rate, iteration, A_ones, b):
    x_list = [root_init]
    m = A_ones.shape[0]
    for i in range(iteration):
        x_new = x_list[-1] - learning_rate * grad(A_ones, b, x_list[-1])

        if (np.linalg.norm(grad(A_ones, b, x_new))/m < 1e-1 ): break

        x_list.append(x_new)

    return x_list

def draw_initially_parabola(root_init, x0):
    y0 = root_init[0][0]*x0*x0 + root_init[1][0]*x0 + root_init[2][0]
    plt.plot(x0, y0, color="black")

#Draw x_list through gradient descent
def draw_parabolas(x_list, x0):
    for i in range(len(x_list)):
        y0 = x_list[i][0][0]*x0*x0 + x_list[i][1][0]*x0 + x_list[i][2][0]
        plt.plot(x0, y0, color="green", alpha=0.3)

def update(i):
    y0 = x_list[i][0][0]*x0*x0 + x_list[i][1][0]*x0 + x_list[i][2][0]
    line.set_data(x0, y0)
    return line ,

if __name__ == "__main__":

	#Data
    A = np.array([[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]]).T
    b = np.array([[2, 5, 7, 9, 11, 16, 19, 23, 22, 29, 29, 35, 37, 40, 46, 42, 39, 31, 30, 28, 20, 15, 10, 6]]).T

    #Munipulate data of A
    A_ones = munipulate_A_data(A)

    #Formula
    x0 = np.linspace(0, 30, 10000)
    x0, y0 = create_parabola(A_ones, b, x0)

    #Gradient descent
    # root_init = np.array([[1.05455731], [3], [-0.01216413]])
    # root_init = np.array([[-2.1], [3.1], [-2.1]])
    root_init = np.array([[-2.1], [5.1], [-2.1]])
    iteration = 999
    learning_rate = 0.000001
    x_list = GD_create_parabolas(root_init, learning_rate, iteration, A_ones, b)
    print(len(x_list))

    #Visualize data and parabolas
    fig, ax = draw_data(A, b, x0, y0)
    draw_initially_parabola(root_init, x0)
    draw_parabolas(x_list, x0)

    plt.legend(("Value in each GD iteration", "Solution by formula ", "Initial value for GD"), loc=(0.52, 0.01))
    ltext = plt.gca().get_legend().get_texts()
    
    #Draw animation
    line , =  ax.plot([], [], color="purple")
    iters = np.arange(1, len(x_list), 1)
    line_animation = animation.FuncAnimation(fig, update, iters, interval=50, blit=True)

plt.show()