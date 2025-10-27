# Multi-Layer Perceptron (MLP) — Classification sur Digits

Ce projet implémente un réseau de neurones fully connected (MLP) pour classer les chiffres manuscrits du dataset `load_digits` de Scikit-learn.

## Objectif
Utiliser un MLP avec régularisation L2 et Dropout pour prédire les chiffres (0 à 9) à partir d’images 8x8 aplaties.

## Ce que j’ai appris
- La structure d’un MLP : couches `Dense`, activations, régularisation
- L’utilisation de `EarlyStopping` et `ModelCheckpoint` pour stabiliser l’entraînement
- Comment visualiser les courbes d’apprentissage (accuracy et loss)

## Structure du code
- Prétraitement des données : normalisation + one-hot encoding
- Définition du modèle avec `Sequential` (128 → Dropout → 64 → Dropout → 10)
- Compilation avec `Adam` et `categorical_crossentropy`
- Entraînement avec callbacks
- Visualisation des courbes d’accuracy et de loss

## Exécution
```bash
pip install -r requirements.txt
python train.py
