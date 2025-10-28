# Deep Learning — Architectures chirurgicales et interprétables

Ce dossier regroupe mes implémentations stratégiques des architectures fondamentales du deep learning. Chaque projet est conçu pour être reproductible, interprétable, et aligné avec ma posture de developpeur GenRL.

---

## Vision

Je ne me contente pas d’utiliser des modèles — je les construis, les régularise, les interprète.  
Chaque réseau ici est une brique neuronale posée avec intention, rigueur et clarté.  
Ce dossier est la transition entre mes fondations ML et mes architectures RLHF.

---

## Projets inclus

| Architecture          | Dataset  | Objectif principal                           | Interprétabilité |
|-----------------------|----------|----------------------------------------------|------------------|
| MLP                   | Digits   | Classification multiclasse avec réseau fully connected | Courbes d’apprentissage |
| CNN classique         | CIFAR-10 | Classification visuelle avec régularisation et augmentation | Activations des filtres |
| MobileNetV2           | CIFAR-10 | Transfer learning + fine-tuning + Grad-CAM | Grad-CAM + activations |

---

## Structure de chaque projet

- `train.py` : script principal
- `README.md` : explication chirurgicale
- `requirements.txt` : dépendances

---

## Exécution

```bash
cd DeepLearning/mlp/
pip install -r requirements.txt
python train.py
