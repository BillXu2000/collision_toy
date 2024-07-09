Y = [0.0, 0.038415782, 0.15366313, 0.3457403, 0.6146525, 0.9603975, 1.3774202, 0.46751934, 0.0, 0.008714974, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
X = list(range(len(Y)))
import matplotlib.pyplot as plt
import numpy as np

X = np.array(X)
Y = np.array(Y)

fig, ax = plt.subplots()
ax.plot(X, Y)
plt.show()
