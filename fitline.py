from matplotlib.backend_bases import MouseButton
import matplotlib.pyplot as plt
import numpy as np
x = np.random.uniform(-2, 5, 10).tolist()
y = np.random.uniform(0,3,10).tolist()

# Linear solver
def my_linfit(x, y):
    # print(x)
    # print(y)
    xy_bar = 0
    for i in range(len(x)):
        xy_bar += x[i] * y[i]
    xy_bar = xy_bar / len(x)
    # print(xy_bar)
    x_bar_y_bar = np.mean(x) * np.mean(y)
    x_square_bar = 0
    for i in range(len(x)):
        x_square_bar += x[i]**2
    x_square_bar = x_square_bar / len(x)
    x_bar_square = np.mean(x)**2
    a = (xy_bar - x_bar_y_bar) / (x_square_bar - x_bar_square)
    b = np.mean(y) - a * np.mean(x)
    return a, b

# Define what onclick controller does
def onclick(event):
    if event.button == MouseButton.LEFT:
        x.append(event.xdata)
        y.append(event.ydata)
        #clear frame
        plt.clf()
        plt.plot(x,y,'kx'); #inform matplotlib of the new data
        plt.draw()
    elif event.button == MouseButton.RIGHT:
        #clear frame
        plt.clf()
        plt.plot(x,y,'kx'); #inform matplotlib of the new data
        # Calculate new a and b for the best fit line
        a, b = my_linfit(x, y)
        xp = np.arange(-2, 5, 0.1)
        plt.plot(xp, a*xp+b, 'r-')
        plt.draw() #redraw

fig,ax=plt.subplots()
# Plot the points that are currently captured
ax.plot(x,y,'kx')
# Calculate the a and b for the best fit line
a, b = my_linfit(x, y)
xp = np.arange(-2, 5, 0.1)
# Plot primary best fit line
ax.plot(xp, a*xp+b, 'r-')
# Set up onclick controller
fig.canvas.mpl_connect('button_press_event',onclick)
plt.show()
plt.draw()