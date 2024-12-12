import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d

SAVE_PATH = "datasets\\RegressionProblem\\at2po30.csv"
NUMBER_OF_FEATURES = 2
POPULATION_SIZE = 100


if __name__ == "__main__":
    x1_values = np.arange(-1, 1, step=0.3)
    x2_values = np.arange(-1, 1, step=0.3)

    def fun(x1, x2):
        w1 = 3.1
        w2 = 2
        b = 10
        return w1*x1 + w2*x2 + b + np.random.randn(1)
    
    data_dim = x1_values.shape[0]
    population_data = np.empty([data_dim * data_dim, 3], dtype=np.float32)
    for i, x1 in enumerate(x1_values):
        for j, x2 in enumerate(x2_values):
            population_data[i * data_dim + j, 1] = x1
            population_data[i * data_dim + j, 2] = x2
            population_data[i * data_dim + j, 0] = fun(x1, x2)    
    
    columns = ["value"] + [ "_" + str(i) for i in range(NUMBER_OF_FEATURES)]
    dataframe = pd.DataFrame(population_data, columns=columns)

    # ax = plt.figure().add_subplot(projection='3d')
    # X, Y, Z = dataframe["_0"].to_numpy(), dataframe["_1"].to_numpy(), dataframe["value"].to_numpy()
    # X, Y = np.meshgrid(X, Y)
    # print(X)
    # ax.contour(X, Y, Z, cmap=cm.coolwarm)  # Plot contour curves
    # ax.scatter(X, Y, Z, marker="o")
    # plt.show()

    print(dataframe)
    dataframe.to_csv(SAVE_PATH, sep=',', header=False, index=False)