import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("loss_landscape.csv", header=None, sep=";")

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(data[0].to_numpy(), data[1].to_numpy(), data[2].to_numpy(), s=0.5)

learning_data = pd.read_csv("learning.csv", header=None, sep=";")
ax.scatter(learning_data[0].to_numpy(), learning_data[1].to_numpy(), learning_data[2].to_numpy(), color='red', s=1)
plt.show()

# print(data)