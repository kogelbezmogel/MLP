import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

SAVE_PATH = "datasets/TwoClassProblem/at2po30.csv"
NUMBER_OF_CLASSES = 2
NUMBER_OF_FEATURES = 2
POPULATION_SIZE = 600

# means = [ np.array(arr) for arr in ([1, 1, 1], [1, 1, -1], [1, -1, -1]) ]
# variances = [ np.array(arr) for arr in ([1, 1, 1], [1, 1, 1], [1, 1, 1]) ]

means = [ np.array(arr) for arr in ([-1.2, -1.2], [1.2, 1.2]) ]
variances = [ np.array(arr) for arr in ([1, 1], [1, 1]) ]

if __name__ == "__main__":

    full_data = np.empty([0, NUMBER_OF_FEATURES+1])

    for class_id in range(0, NUMBER_OF_CLASSES):

        # creating labels for population
        target = (class_id) * np.ones([POPULATION_SIZE, 1])

        # creating class_i population
        population_i = np.random.randn(POPULATION_SIZE, NUMBER_OF_FEATURES)
        
        # adding mean and variance
        population_i = population_i * variances[class_id] + means[class_id]
        population_data = np.concatenate((target, population_i), axis=1)
        full_data = np.concatenate([full_data, population_data], axis=0)
    
    columns = ["label"] + [ "_" + str(i) for i in range(NUMBER_OF_FEATURES)]
    dataframe = pd.DataFrame(full_data, columns=columns)
    dataframe.to_csv(SAVE_PATH, sep=',', header=False, index=False)