import pandas as pd

if __name__ == '__main__':
    data = pd.read_csv('datasets//Iris//iris.csv', sep=',')
    labels = data['Species'].unique()

    for i, label in enumerate(labels):
        data.loc[data['Species'] == label, "Species"] = i
    
    data = data[["Species", "SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
    data.to_csv('datasets//Iris//iris.csv', sep=',', index=False)
