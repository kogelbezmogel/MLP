import PIL
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from functools import cmp_to_key

LOSS_SPACE_FILE = "loss_landscape.csv"
LEARNING_FILE = "learning.csv"


def create_image_sequence(learn_data, space_data):
    for i, row in learn_data.iterrows():
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(space_data[0].to_numpy(), space_data[1].to_numpy(), space_data[2].to_numpy(), s=0.3)
        ax.plot(learn_data[0].iloc[: i].to_numpy(), learn_data[1].iloc[: i].to_numpy(), learn_data[2].iloc[: i].to_numpy(), color='red', linewidth=0.2, markersize=0.5, marker='o')

        plt.savefig(f"./gif_plots/plot_{i}")
        ax.clear()
        plt.close()




if __name__ == "__main__":
    space_data = pd.read_csv(LOSS_SPACE_FILE, header=None, sep=";")
    learn_data = pd.read_csv(LEARNING_FILE, header=None, sep=";")

    create_image_sequence(learn_data, space_data)

    image_paths = [os.path.join("gif_plots", file) for file in os.listdir("./gif_plots")]
    image_paths = sorted( image_paths, key = cmp_to_key( lambda x1, x2: int(x1.split('_')[2].split('.')[0]) - int(x2.split('_')[2].split('.')[0]) ) )
    images = [PIL.Image.open(image_path) for image_path in image_paths]

    images[0].save(
        "./learning.gif",
        save_all = True,
        append_images = images[1:],
        duration = 300,
        loop = 0
    )

    



