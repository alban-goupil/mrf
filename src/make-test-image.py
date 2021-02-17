import numpy as np
from scipy.stats import norm 

## Génération d'exemple d'images. Ces images sont obtenus
## via des observations d'un ensemble de pixels classés
## selon des labels.

## * Paramètres
width, height = 50, 50                # Taille de l'image
labelsfilename = '../data/labels.npy' # Fichier des labels
datafilename = '../data/data.npy'     # Fichier des données

## mus et sigmas désignent les lois
## Pr[observervations|label]. Ces dernières sont ici des
## gaussiennes. Par exemple Pr[obs | label=3] = N(mus[3],
## sigmas[3])
mus = np.array([40, 80, 120, 160, 200])  # Centre de classes
sigmas = np.array([10, 10, 10, 10, 10])  # Leur ecart-types


## * Générations des labels

## L'image est décomposé en 5 parties: l'est, le nord,
## l'ouest, le sud et le centre.
labels = 5*np.ones((height, width), dtype=np.int)
ys, xs = np.meshgrid(np.linspace(-1, 1, height),
                     np.linspace(-1, 1, width))
labels[np.abs(xs) <= ys] = 0
labels[np.abs(xs) <= -ys] = 2
labels[np.abs(ys) <= -xs] = 1
labels[np.abs(ys) <= xs] = 3
labels[xs**2 + ys**2 < 0.5] = 4

np.save(labelsfilename, labels)


## * Générations des observations

## Si un pixel appartient à la classe c alors l'observation
## associée est une réalisation d'une loi normale de
## moyennes mus[c] et d'écart-type sigmas[c].
data = norm.rvs(loc=mus[labels], scale=sigmas[labels])
np.save(datafilename, data)
