import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits import mplot3d
from pathlib import Path
from os import path, listdir
from os.path import isfile, join
from PIL import Image, ImageFile
import imageio

ImageFile.LOAD_TRUNCATED_IMAGES = True

dir_path = path.dirname(path.realpath(__file__))


def make_gif_from_frames():
    fp = path.join(dir_path, 'figures/gif/')
    fp_out = path.join(dir_path, 'figures/movie.gif')
    filenames = [f for f in listdir(fp) if isfile(join(fp, f))]
    for i in range(len(filenames)):
        filenames[i] = filenames[i][:-4]
    filenames.sort(key=int)
    images = []
    for filename in filenames:
        images.append(imageio.imread(path.join(dir_path, 'figures/gif/{}.png'.format(filename))))
    imageio.mimsave(fp_out, images, fps=30)

results = pd.read_csv('IRF4_fitting_individuals_gen50_size50000.csv')
results = results.iloc[0:10000]
best_result = results.iloc[results['fitness'].idxmin()]
# results = results.loc[results['fitness']<20]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scat = ax.scatter3D(results['beta'], results['p'], zs=results['fitness'], c=results['fitness'], cmap='hot', \
                                                                                                       marker='.')
scat_best = ax.scatter3D(best_result['beta'], best_result['p'], zs=best_result['fitness'], c='green', marker='o',
                         s=20)
ax.set_xlabel('beta', fontweight='bold')
ax.set_ylabel('p', fontweight='bold')
ax.set_zlabel('fitness', fontweight='bold')
ax.set_xlim([0, 20])
ax.set_ylim([0, 0.08])
ax.set_zlim([0, 12])
fig.colorbar(scat, ax=ax)
plt.title("Fitnesses in Solution Space")

for angle in np.arange(0, 360, 1):
    ax.view_init(15, angle)
    plt.savefig(path.join(dir_path, 'figures/gif/{}'.format(angle)))

make_gif_from_frames()