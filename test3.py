import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(0, 10*np.pi, 1000)
Y = np.sin(X)

fig, ax = plt.subplots()
ax.plot(X, Y)
plt.show()
