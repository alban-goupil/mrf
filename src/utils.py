import numpy as np
from numpy.linalg import eigh, norm
from numba import njit


#@jit(nopython=True)
def negnormlogpdf(Y, mu, sigma, tol=1e-5):
  """Retourne la logpdf d'une loi normale multivariée de
  moyenne mu et de covariance sigma. Fonctionne aussi si
  sigma est singulière en projetant les données sur l'espace
  signal.
  """
  w, v = eigh(sigma)        # Diagonalisation
  idx = w > tol             # Indices des valeurs propres >> 0
  rk = np.sum(idx)          # Le rang
  logdet = np.sum(np.log(w[idx])) # Déterminant non dégénérée

  # Le projecteur
  iw = np.zeros_like(w) # Inverse de w pour la partie non dégénérée
  iw[idx] = w[idx] ** -0.5
  P = v.T @ np.diag(iw)

  # Maintenant D devient quasiment standard
  if Y.ndim == 1:
    D = (Y.reshape(-1, 1) - mu) @ P
  else:
    D = (Y - mu) @ P

  # Il faut aussi prendre en compte le Jacobien de la
  # transformation linéaire
  return 0.9189385332046727 * rk + 0.5 * np.sum(D**2, axis=1) + 0.5 * logdet



def neighborhood(height, width):
  """Retourne le voisinage de Von Neumann pour chaque pixel
  d'une image de taille `height` par `width`. Celui-ci est
  donné par la liste des paires des indices des pixels
  voisins. Chaque arrête appraît deux fois (a, b) et (b, a).
  """
  neighbors = []
  for k in range(height * width):
    i, j = divmod(k, width)
    neighbors.append([])
    if 0 <= i-1: neighbors[-1].append(k - width)       # Au dessus
    if i + 1 < height: neighbors[-1].append(k + width) # En dessous
    if 0 <= j-1: neighbors[-1].append(k - 1)           # À gauche
    if j + 1 < width: neighbors[-1].append(k + 1)      # À droite
    neighbors[-1].sort()

  rows = np.cumsum([0] + list(map(len, neighbors)))
  cols = np.concatenate(neighbors)
  
  return neighbors, rows, cols



## * Energie et Gibbs sampling
@njit
def energy(pis, phis, psis, rows, cols, X):
  """Calcul l'énergie des étiquettes X étant donné le
  modèle."""
  U = 0
  for i, xi in enumerate(X):
    U += pis[xi] + phis[i, xi]
    for j in range(rows[i], rows[i+1]):
      U += psis[xi, X[cols[j]]] / 2
  return U
