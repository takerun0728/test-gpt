from sklearn.datasets import load_digits
import numpy as np
import matplotlib.pyplot as plt

digit = load_digits()
plt.matshow(digit.images[0])
print(np.allclose(digit.images, digit.data.reshape(-1, 8, 8)))

plt.show()