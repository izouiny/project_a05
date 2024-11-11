# Notes taken during the project

## 1. Features engineering

### Ajout des features

- Distance du filet : Déjà fait durant le milestone 1
- Angle relatif au filet : Déjà fait durant le milestone 1
- Est un but (0 ou 1) : Fait (colonne `is_goal`)
- Filet vide : Fait (colonne `is_empty_net`)

### Split des données

J'ai ajouté une fonction `load_train_test_dataframes` pour séparer les donnes en train et test sets.
Cette methode retourne deux dataframes :

- train: saisons 2016, 2017, 2018 et 2019
- test: saison 2020, sans la colonne `is_goal`
