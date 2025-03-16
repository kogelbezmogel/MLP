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
        fig = plt.figure( figsize=(10, 10) )
        ax = fig.add_subplot(projection='3d')
        ax.view_init(elev=5, azim=85)
        ax.scatter(space_data[0].to_numpy(), space_data[1].to_numpy(), space_data[2].to_numpy(), s=1, alpha=0.1)
        ax.plot(learn_data[0].iloc[: i].to_numpy(), learn_data[1].iloc[: i].to_numpy(), learn_data[2].iloc[: i].to_numpy(), color='red', linewidth=0.5, markersize=1, marker='o')
        ax.axis("off")

        plt.savefig(f"./gif_plots/plot_{i}")
        ax.clear()
        plt.close()




if __name__ == "__main__":
    space_data = pd.read_csv(LOSS_SPACE_FILE, header=None, sep=";")
    learn_data = pd.read_csv(LEARNING_FILE, header=None, sep=";")

    # create_image_sequence(learn_data, space_data)

    image_paths = [os.path.join("gif_plots", file) for file in os.listdir("./gif_plots")]
    image_paths = sorted( image_paths, key = cmp_to_key( lambda x1, x2: int(x1.split('_')[2].split('.')[0]) - int(x2.split('_')[2].split('.')[0]) ) )
    
    images = [PIL.Image.open(image_path) for image_path in image_paths]
    img_width, img_heght = images[0].size
    edge = 200
    images = [ image.crop([edge, edge, img_width-edge, img_heght-edge]) for image in images ]

    images[0].save(
        "./learning_classification_sgd.gif",
        save_all = True,
        append_images = images[1:300],
        duration = 50,
        loop = 0
    )

    # fig = plt.figure( figsize=(10, 10) )
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(space_data[0].to_numpy(), space_data[1].to_numpy(), space_data[2].to_numpy(), s=1, alpha=0.1)
    # ax.plot(learn_data[0].to_numpy(), learn_data[1].to_numpy(), learn_data[2].to_numpy(), color='red', linewidth=0.5, markersize=1, marker='o')
    # # ax.axis("off")
    # plt.show()



