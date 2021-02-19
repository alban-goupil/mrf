import time
import numpy as np
import numpy.random
from scipy.stats import norm
from utils import neighborhood, energy
import matplotlib.pyplot as plt
import numba

## * Paramètres
# Fichiers de données
filename = '../data/data.npy'      # Fichier d'observations
lblfilename = '../data/labels.npy' # Fichier de labels pour vérif.

# Param. des observations
mus = np.array([40., 80., 120., 160., 200.]) # Centre des classes
sigmas = np.array([10., 10., 10., 10., 10.]) # Leur écart-type

# Mean Field Theory
niter = 200              # Nombre d'itérations
tol = 1e-8               # Tolérance sur la variation totale

rng = numpy.random.default_rng() # Pseudo-source


## * Observations
Y = np.load(filename)           # Observations
T = np.load(lblfilename)        # Valeur terrain

nclasses = 1 + int(np.max(T))   # Nombre de classes
nobs = int(Y.size)              # Nombre d'observations
height, width = Y.shape         # Taille de l'image
Y = np.ravel(Y)                 # Linéarisation



## * Modèle par MRF                   
# A priori par pixel
pis = np.log(nobs) - np.log(np.bincount(T.ravel()))

# Vraisemblance de chaque pixel
phis = np.empty((nobs, nclasses))
for c in range(nclasses):
  phis[:, c] = -norm.logpdf(Y, mus[c], sigmas[c])

# A priori des voisins
psis = -2.0 * np.eye(nclasses)

# Voisinage: Von Neumann
edges, rows, cols = neighborhood(height, width)



## * Mean Field Theory
Xml = np.argmin(phis, axis=-1)  # Maximum de vraisemblance
Xmap = Xml.copy()               # Meilleur X jusqu'à présent
Umap = energy(pis, phis, psis, rows, cols, Xmap) # et son énergie

beliefs = np.ones_like(phis) / nclasses
Us = [Umap]                     # Historique des énergies

tic = time.time()
# Recherche de point fixe par itération
for k in range(niter):
  print (k, '/', niter, end='\r')

  # Amélioration de l'estimation
  oldbeliefs = beliefs.copy()
  bjs = oldbeliefs @ psis / 2
  U = pis + phis
  for i in range(nobs):
    for j in range(rows[i], rows[i+1]):
      U[i] += bjs[cols[j]]
  beliefs = np.exp(-U)
  beliefs /= np.sum(beliefs, axis=1, keepdims=True)

  # On la garde si c'est le meilleur
  X = np.argmax(beliefs, axis=-1)  # Maximum de vraisemblance
  Us.append(energy(pis, phis, psis, rows, cols, X))
  if Us[-1] < Umap:
    Umap, Xmap = Us[-1], X.copy()

  # On s'arrête si rien ne bouge
  if np.allclose(beliefs, oldbeliefs, tol):
    break
tac = time.time()



## * Affichage
print(f'Temps: {tac - tic}')
print(f'U Truth = {energy(pis, phis, psis, rows, cols, T.ravel()):10.2f}')
print(f'U ML    = {Us[0]:10.2f}\t#err = {np.sum(T.ravel() != Xml)}')
print(f'U MAP   = {Umap:10.2f}\t#err = {np.sum(T.ravel() != Xmap)}')

plt.close('all')
plt.figure(1)
plt.subplot(221)
plt.imshow(T)
plt.title('Vérité')

plt.subplot(222)
plt.imshow(Y.reshape((height, width)))
plt.title('Observations')

plt.subplot(223)
plt.imshow(Xml.reshape((height, width)))
plt.title('Maximum de vraisemblance')

plt.subplot(224)
plt.imshow(Xmap.reshape((height, width)))
plt.title('Maximum a posteriori estimé par MFT')


plt.figure(2)
plt.clf()
plt.plot(Us)

plt.show(block=False)
