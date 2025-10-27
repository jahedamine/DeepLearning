# MobileNetV2 — Classification CIFAR-10 avec Fine-Tuning et Interprétabilité

Ce projet implémente un modèle MobileNetV2 pré-entraîné pour classer les images du dataset CIFAR-10, avec fine-tuning, visualisation des activations et interprétation Grad-CAM.

## Objectif
Utiliser un backbone MobileNetV2 pré-entraîné sur ImageNet pour effectuer une classification multiclasse sur CIFAR-10, tout en explorant les activations internes et les zones d’attention du modèle.

## Ce que j’ai appris
- Comment adapter un modèle pré-entraîné à un nouveau dataset (transfer learning)
- L’importance du fine-tuning partiel pour améliorer la performance
- Comment visualiser les activations des filtres convolutifs
- L’utilisation de Grad-CAM pour interpréter les décisions du modèle

## Structure du code
- Chargement et redimensionnement du dataset CIFAR-10 (96x96)
- Définition du modèle :
  - MobileNetV2 sans la tête (`include_top=False`)
  - GlobalAveragePooling → Dense → Dropout → Softmax
- Entraînement initial avec backbone gelé
- Fine-tuning partiel (dégel des couches supérieures)
- Évaluation : accuracy, matrice de confusion, classification report, F1-score
- Visualisation :
  - Courbes d’apprentissage
  - Activations des filtres convolutifs
  - Grad-CAM sur une image test

## Exécution
```bash
pip install -r requirements.txt
python train.py
