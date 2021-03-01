import numpy as np
import matplotlib.pyplot as plt
 
def sigmoid(x, b=0):
    f = -(x + b)
    return 1 / (1 +np.exp(f))
 
x = np.arange(-10, 10, 0.1)
y = sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()