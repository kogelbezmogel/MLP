import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data_path = "datasets/TwoClassProblem/at2po30.csv"

data = pd.read_csv(data_path, header=None)

labels = data[0].unique().astype(np.int32)

for label in labels:
    plt.scatter(data[data[0] == label][1], data[data[0] == label][2], alpha=0.5, s=5)
plt.show()