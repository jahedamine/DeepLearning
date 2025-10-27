# Convolutional Neural Network (CNN) — Classification CIFAR-10

Ce projet implémente un CNN classique pour classer les images du dataset CIFAR-10 en 10 catégories visuelles.

## Objectif
Utiliser un réseau convolutif profond avec régularisation et augmentation de données pour prédire la classe d’objets dans des images couleur 32x32.

## Ce que j’ai appris
- La structure d’un CNN : convolution, pooling, batchnorm, dropout
- L’impact de l’augmentation de données sur la généralisation
- L’utilisation de callbacks pour stabiliser l’entraînement
- Comment visualiser les activations des filtres convolutifs

## Structure du code
- Chargement et normalisation du dataset CIFAR-10
- Définition du modèle avec `Sequential` :
  - 3 blocs Conv2D + MaxPooling + BatchNorm + Dropout
  - Flatten → Dense → Dropout → Softmax
- Compilation avec `Adam` et `sparse_categorical_crossentropy`
- Entraînement avec `ImageDataGenerator` et callbacks
- Évaluation : accuracy, matrice de confusion, classification report
- Visualisation : courbes d’apprentissage + activations des filtres

## Exécution
```bash
pip install -r requirements.txt
python train.py
