# Deep Learning — Architectures chirurgicales et interprétables

Ce dossier regroupe mes implémentations stratégiques des architectures fondamentales du Deep Learning.
Chaque modèle est construit, entraîné et interprété avec rigueur, en gardant une vision d’ingénieur orientée :
  modularité,
  explainability,
  scalabilité,
  préparation à l’industrialisation.  
C’est l’étape clé entre mes algorithmes ML classiques et mes travaux avancés en RL, RLHF, DPO, Agents GenAI et Alignement.

---

## Vision

Je ne me limite pas à entraîner des modèles.
Je développe des architectures maîtrisées de bout en bout, avec un focus sur :
  la régularisation,
  les choix d’optimisation,
  l’interprétation des features apprises,
  la reproductibilité,
  et la performance réelle.
Mon objectif est simple :
  créer des modèles qui apprennent, qui généralisent, et qui peuvent être déployés.

---

## Projets inclus

| Architecture          | Dataset  | Objectif principal                           | Interprétabilité |
|-----------------------|----------|----------------------------------------------|------------------|
| MLP                   | Digits   | Classification multiclasse avec réseau fully connected | Courbes d’apprentissage |
| CNN classique         | CIFAR-10 | Classification visuelle avec régularisation et augmentation | Activations des filtres |
| MobileNetV2           | CIFAR-10 | Transfer learning + fine-tuning + Grad-CAM | Grad-CAM + activations |

---
## Pourquoi ces architectures ?

 MLP : compréhension des réseaux de base + surapprentissage + tuning
 CNN : fondamentaux vision + convolution + pooling + régularisation
 MobileNetV2 : modèle léger → production ready et idéal pour mobile/edge AI
 Grad-CAM : interprétation → critère essentiel pour IA fiable et alignée
 Fine-tuning : compétence clé pour le métier ML / GenAI Engineer

---
## Structure de chaque projet

DeepLearning/
│── mlp/
│   ├── train.py
│   ├── README.md
│   ├── requirements.txt
│
│── cnn/
│   ├── train.py
│   ├── README.md
│   ├── requirements.txt
│
│── mobilenet/
    ├── train.py
    ├── README.md
    ├── requirements.txt
    
---
## Applications industrielles

Ces architectures sont utilisées dans :
  classification d’images (médical, retail, sécurité)
  détection d’anomalies visuelles
  optimisation des modèles pour mobile (edge AI)
  prétraitement visuel avant RL
  extraction de features pour agents intelligents

---
## Exécution

```bash
cd DeepLearning/mlp/
pip install -r requirements.txt
python train.py
---
## Auteur

Made by Amine Jahed
GenAI & ML Engineer — Agents modulaires, Alignment, RL/RLHF.
Vision : construire des modèles compréhensibles, efficaces et prêts pour la production, du prototype jusqu’à l’IA alignée.
