# PKLot — détection de place libre / occupée par **deep learning (CNN)**

Ce dossier **s'ajoute** au projet sans modifier les versions précédentes (`parking_v5.py` et le reste du dépôt ne sont pas touchés).
Il classe une image de place de parking en **2 classes** — `empty` (free) / `occupied` (busy) — avec un réseau de neurones
convolutif (Keras / TensorFlow), puis le **compare aux méthodes LBP + 1-NN (distance SAD)** existantes, en particulier la
**pyramide LBP 1-2-4**.

## Contenu

| Fichier | Rôle |
|---|---|
| `PKLOT_deep_learning.ipynb` | notebook principal (CNN, comparaison, graphiques). Livré **avec ses sorties** |
| `parking_v6.py` | copie du pipeline LBP de la branche `parking-v6` (LBP, LBP uniforme, histogrammes par blocs, chronomètre), utilisée par le notebook. Deux modifications : chemin des données relatif (`../Database`) et multi-échelle `GRID_SCALES = [2]` (2×2 = 4 histogrammes concaténés, 236 valeurs) |
| `resultats_dl/` | graphiques (`.png`), tableaux (`.csv`) et résultats par tirage (`tirage_*.json`) |

Ne sont **pas** versionnés (regénérés au premier lancement, cf. `.gitignore`) : `resultats_dl/cache_*.npz` (images et
caractéristiques en cache) et `resultats_dl/*.keras` (modèles entraînés).

## Protocole

- Dataset PKLot : `Database/A/{busy,free}` et `Database/B/{busy,free}`, combinés par classe (4182 `empty`, 8402 `occupied`).
  Le dossier `Database/` est retrouvé automatiquement (variable d'environnement `PKLOT_DATA_DIR` pour le forcer).
- **5 tirages aléatoires** (seeds `42, 1, 2, 3, 4` = `CV_SEEDS` de l'ancien code). À chaque tirage : **500 images/classe de test**
  (jamais dans le train), **100 images/classe d'entraînement**, 20 images/classe de validation (early stopping du CNN).
  Les images de test sont les mêmes que celles de l'ancien pipeline pour une seed donnée.
- **Le CNN n'est entraîné qu'une fois par tirage** (5 entraînements) et le résultat est mis en cache : relancer le notebook ne
  réentraîne rien (`FORCE_RETRAIN = True` pour forcer). Images et caractéristiques LBP sont lues/calculées une seule fois.
- Deux protocoles comparés, sur **exactement les mêmes tirages et le même test** :
  1. **100 img/classe** en entraînement pour tout le monde (même budget) ;
  2. **protocole complet** : pyramide LBP sur 3682 img/classe (ancien protocole) ; CNN sur 3482 img/classe + 200 img/classe de
     validation prises dans ces mêmes 3682 images.
- **CNN** (mêmes briques que le notebook `INTRO_Cyber`) : entrée RGB 64×64, `Rescaling(1/255)`, 3 × [Conv2D 3×3 (32→64→128) + ReLU +
  MaxPooling], Dropout 0.25, Dense 256 (Dropout 0.5), Dense 50, **Dense 2 (softmax)**, Adam,
  `sparse_categorical_crossentropy`, EarlyStopping sur `val_loss` (patience 5, meilleurs poids restaurés), 30 epochs max.
- **Méthodes classiques** : histogramme LBP (gris / mosaïque RGB / un LBP par plan / multi-échelle 2×2 / pyramide 1-2-4),
  classification par 1-plus-proche-voisin avec la distance SAD. Les vecteurs sont vérifiés identiques à ceux de `parking_v6.py`.

## Résultats (moyenne de 5 tirages, 1000 images de test)

| Méthode | 100 img/classe | Protocole complet (3682 img/classe) |
|---|---|---|
| **CNN (2 classes)** | 90.86 % (± 2.56) | 99.48 % (± 0.17) |
| LBP gris | 91.34 % | 99.62 % |
| LBP mosaïque RGB (v5) | 91.52 % | 99.66 % |
| LBP par plan RGB | 91.52 % | 99.74 % |
| LBP multi-échelle 2×2 (4 histogrammes) | 91.86 % | 99.86 % |
| **LBP pyramide 1-2-4** | 92.66 % (± 1.23) | 99.92 % (± 0.10) |

**CNN vs pyramide 1-2-4** : à 100 img/classe la pyramide est en moyenne 1.8 point au-dessus (le CNN gagne 3 tirages sur 5 mais
est très variable : 86.4 % à 93.3 %) ; avec le protocole complet l'écart tombe à 0.44 point en faveur de la pyramide (elle gagne
les 5 tirages). Sur PKLot, le CNN n'apporte pas de gain par rapport à la pyramide LBP, et il est beaucoup plus lent
(≈ 36–54 s d'entraînement par tirage contre ≈ 2 s pour le 1-NN).

Graphiques : `resultats_dl/comparaison_moyenne_100.png`, `comparaison_cnn_anciennes_methodes.png`, `cnn_vs_pyramide.png`,
`cnn_courbes_confusion.png` ; tableaux : `comparaison_tirages.csv`, `cnn_vs_pyramide.csv`.

![CNN vs pyramide](resultats_dl/cnn_vs_pyramide.png)

## Utilisation

```bash
pip install tensorflow pandas scikit-learn matplotlib pillow jupyter
jupyter notebook deep_learning/PKLOT_deep_learning.ipynb
```

Premier lancement : lecture des 12 584 images et entraînement des CNN (environ 5 min sur CPU). Lancements suivants : ≈ 20 s
(tout vient du cache). TensorFlow ≥ 2.11 n'utilise pas le GPU sous Windows natif — le CPU suffit ici.

## Limites

Les images d'un même parking se ressemblent beaucoup (même place, jours différents) : le tirage aléatoire train/test met des
vues quasi identiques des deux côtés, ce qui gonfle les accuracies (surtout > 99 %). Les comparaisons relatives restent valables
car toutes les méthodes utilisent les mêmes tirages.
