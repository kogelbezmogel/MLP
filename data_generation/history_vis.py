import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    history = pd.read_csv("iris_800e.csv", header=0, sep=',')
    
    history.drop('epoch', axis=1, inplace=True)
    print(history.columns)
    plt.plot(history[['train_acc', 'test_acc']] * 100)
    plt.legend(history.columns[2:])
    plt.title("Accuracy during trainig")
    plt.xlabel("epochs")
    plt.ylabel("accuracy (%)")
    plt.show()