# Notes taken during the project

## 1. Features engineering

### Ajout des features

- Distance du filet : Déjà fait durant le milestone 1
- Angle relatif au filet : Déjà fait durant le milestone 1
- Est un but (0 ou 1) : Fait (colonne `is_goal`)
- Filet vide : Fait (colonne `is_empty_net`)

#### Correction goal_angle

Correction de la feature `goal_angle` from -180 (left) to 180 (right).
Les valeurs negatives correspondent à un angle à gauche du point de vue du gardien de but.
Les valeurs positives correspondent à un angle à droite du point de vue du gardien de but.
La valeur 0 correspond à un tir en face du gardien de but.
Les valeurs plus grandes que 90 ou plus petites que -90 correspondent à un angle derrière le gardien de but.

![player_to_goal.png](player_to_goal.png)

### Split des données

J'ai ajouté une fonction `load_train_test_dataframes` pour séparer les donnes en train et test sets.
Cette methode retourne deux dataframes :

- train: saisons 2016, 2017, 2018 et 2019
- test: saison 2020, sans la colonne `is_goal`


## 4. Ingénierie des caractéristiques II

### Récupération de tous les types d'événements

J'ai ajouté un argument dans les fonctions `load_events_records`, `load_events_dataframe` et `load_train_test_dataframes`
pour récupérer tous les types d'événements.
Par défaut, ces fonctions retournent seulement les événements de type `shot-on-goal` et `goal` (comme initialement).

Pour récupérer les données séparées en train et test sets, il suffit d'appeler la fonction `load_train_test_dataframes`
avec l'argument `all_types=True`.

```python
from ift6758.data import load_train_test_dataframes
train_data, test_data = load_train_test_dataframes(all_types=True)
```

### Calcul des power-plays

J'ai ajouté la colonne `details_type_code` au DataFrame des événements.
Ceci permettra de calculer les durées des power-plays.

### Ajout d'une méthode permettant de charger les données augmentées

J'ai ajouté les fonctions `load_advanced_dataframe` et `load_advanced_train_test_dataframes` pour charger
les données telles que décrit dans le notebook [feature_engineering_ii.ipynb](notebooks/feature_engineering_ii.ipynb).

Pour charger un Dataframe d'une saison donnée :

```python
from ift6758.features import load_advanced_dataframe
df = load_advanced_dataframe(2016)
```

Pour charger les données séparées en train et test sets :

```python
from ift6758.features import load_advanced_train_test_dataframes
train_data, test_data = load_advanced_train_test_dataframes()
```

### Data preparation for models

J'ai ajouté un pipeline Scikit-learn pour préparer les données avant de les envoyer dans les modèles.
J'ai fait cela avec un pipeline scikit-learn afin que l'on puisse facilement l'utiliser dans un autre pipeline.

Utilisation dans un pipeline Scikit-learn :

```python
from ift6758.features import get_preprocessing_pipeline
from sklearn.pipeline import make_pipeline

preprocessing_pipeline = get_preprocessing_pipeline()
pipeline = make_pipeline(preprocessing_pipeline, model)
```

Utilisation sans pipeline Scikit-learn :

```python
from ift6758.features import get_preprocessing_pipeline

preprocessing_pipeline = get_preprocessing_pipeline()
X_train = preprocessing_pipeline.fit_transform(X_train)
X_test = preprocessing_pipeline.transform(X_test)
```

### Split et nettoyage des données

Ajout d'une fonction `load_train_val_test_x_y` pour charger les données séparées en train, validation et test sets.

```python
from ift6758.features import load_train_val_test_x_y
X_train, y_train, X_val, y_val, X_test, y_test = load_train_val_test_x_y(test_size=0.2)
```