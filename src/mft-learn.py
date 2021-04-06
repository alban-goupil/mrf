import time
import numpy as np
import numpy.random
from scipy.stats import norm
from scipy.special import digamma
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
niter = 21               # Nombre d'itérations
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
pis = np.zeros(nclasses)

# Vraisemblance de chaque pixel
phis = np.empty((nobs, nclasses))
for c in range(nclasses):
  phis[:, c] = -norm.logpdf(Y, mus[c], sigmas[c])

# A priori des voisins
psis = np.zeros((nclasses, nclasses))

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
  oldbeliefs = beliefs.copy()
  oldpis = pis.copy()
  oldpsis = psis.copy()
  
  # Amélioration de l'estimation sur les étiquettes
  bjs = oldbeliefs @ psis / 2
  U = pis + phis
  for i in range(nobs):
    for j in range(rows[i], rows[i+1]):
      U[i] += bjs[cols[j]]
  beliefs = np.exp(-U)
  beliefs /= np.sum(beliefs, axis=1, keepdims=True)

  # Amélioration de pi
  Nks = np.sum(oldbeliefs, axis=0)
  pis = digamma(Nks.sum()-1) - digamma(Nks-1)
  pis -= pis.min()

  # Amélioration de psi
  Nkcs = np.zeros((nclasses, nclasses))
  for i in range(nobs):
    for j in range(rows[i], rows[i+1]):
      Nkcs += 1/2 * np.outer(oldbeliefs[i,:],
                             oldbeliefs[cols[j],:])
  psis = digamma(Nkcs.sum()-1) - digamma(Nkcs-1)
  psis -= psis.min()

  if not (np.isfinite(beliefs).all() and
          np.isfinite(pis).all() and
          np.isfinite(psis).all()):
    beliefs = oldbeliefs
    pis = oldpis
    psis = oldpsis
    print('\nOops')
    break
    
tac = time.time()
Xmap = X.copy()


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


plt.show(block=False)
