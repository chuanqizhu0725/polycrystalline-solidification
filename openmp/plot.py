import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("phi.dat", header=None)
arr = df.values
mat = arr.reshape(1024, 1024)
plt.figure(figsize=(5, 5))
plt.matshow(mat)
plt.show()
