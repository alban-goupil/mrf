# Markov Random Field

## Objectifs

Les modèles d'images basés sur les champs de Markov peuvent
être utilisés pour la segmentation. Ce projet utilise un tel
modèle pour segmenter des images simples selon les méthodes
* du recuit simulé;
* du champ moyen.

## Les programmes

Les sources des programmes sont dans le répertoire
[src/](./src/) et les données dans le répertoire
[data/](./data/).

La librairie [utils.py](./src/utils.py) contient quelques
fonctions utiles à l'ensemble du projet.

Le programme [make-test-image.py](./src/make-test-image.py)
génère des données de test sous forme d'une matrice
d'étiquettes et d'une matrice de valeurs réelles. Une
réalisation de ce programme se trouve dans les fichiers
[labels.npy](./data/labels.npy) pour les étiquettes et
[data.py](./data/data.npy) pour les observations.

Le programme [anneal.py](./src/anneal.py) utilise le recuit
simulé pour estimer les étiquettes à partir des paramètres
du modèle.

Le programme [mft.py](./src/mft.py) utilise la méthode du
champ moyen pour estimer les étiquettes à partir des
paramètres du modèle.

La librairie [utils.py](./src/utils.py) implémente des
fonctions de base comme le calcul d'énergie d'un étiquetage
selon le modèle ou le calcul des pixels voisins d'une grille
donnée.
