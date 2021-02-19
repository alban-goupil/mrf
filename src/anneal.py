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

# Paramètres des observations
mus = np.array([40., 80., 120., 160., 200.]) # Centre des classes
sigmas = np.array([10., 10., 10., 10., 10.]) # Leur écart-type

# Recuit simulé
T0 = 6                          # Température initiale
Trate = 0.99                    # Taux de décroissance
kmax = 200                      # Nombre d'itérations
Ts = T0 * np.power(Trate, range(kmax)) # Températures

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



## * Gibbs sampling
@numba.njit
def gibbscond(pis, phis, psis, rows, cols, Temp, X, i):
  """Retourne les probabilités conditionnelles de X[i] sachant
  X[~i] étant donné le modèle."""
  dU = pis[:] + phis[i, :]
  for j in range(rows[i], rows[i+1]):
    dU += psis[:, X[cols[j]]] / 2
      
  p = np.exp(-dU / Temp) + 1e-20
  return p / p.sum()


def gibbs(pis, phis, psis, rows, cols, Temp, X):
  """Déplace X vers un nouvel échantillon par un
  échantillonnage de Gibbs à la température Temp."""
  for i in range(X.size):
    p = gibbscond(pis, phis, psis, rows, cols, Temp, X, i)
    X[i] = rng.choice(p.size, p=p)



## * Recuit simulé, calcul a posteriori
Xml = np.argmin(phis, axis=-1)  # Maximum de vraisemblance
Xmap = Xml.copy()               # Meilleur X jusqu'à présent
Umap = energy(pis, phis, psis, rows, cols, Xmap) # et son énergie

X = Xmap.copy()                 # État courant
Us = [Umap]                     # L'historique des énergies

tic = time.time()
for k, Tk in enumerate(Ts):
  print (k, '/', Ts.size, end='\r')

  # Échantillonne un nouvel étiquetage
  gibbs(pis, phis, psis, rows, cols, Tk, X)
  Us.append(energy(pis, phis, psis, rows, cols, X))

  # Le garde si c'est le meilleur
  if Us[-1] < Umap:
    Umap, Xmap = Us[-1], X.copy()
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
plt.title('Maximum a posteriori estimé par recuit simulé')


plt.figure(2)
plt.clf()
plt.plot(Us)

plt.show(block=False)
