# Suivi de Pose OSC (YOLOv8)

Ce projet est une démonstration technique réalisée dans le cadre du cours **MUS3329X - Projets en informatique musicale** à la Faculté de musique de l'Université de Montréal.

Il illustre comment utiliser la vision par ordinateur (Computer Vision) pour contrôler des paramètres musicaux ou visuels en temps réel.

## Description

Le script Python capture le flux vidéo d'une webcam, utilise le modèle d'intelligence artificielle **YOLOv8-pose** pour détecter le squelette humain, et transmet les coordonnées des points du corps (nez, poignets, etc.) via le protocole **OSC** (Open Sound Control).

## Fonctionnalités actuelles

*   **Acquisition vidéo** : Lecture fluide de la webcam.
*   **Détection de pose** : Utilisation de YOLOv8 Nano pour une inférence rapide (compatible GPU via CUDA ou MPS).
*   **Communication OSC** : Envoi des coordonnées normalisées (x, y) vers des logiciels comme Max/MSP, PureData, TouchDesigner ou Ableton Live.

## Prérequis

*   Python 3.8 ou plus récent
*   Une webcam fonctionnelle

## Installation

Il est recommandé d'utiliser un environnement virtuel. Installez les dépendances nécessaires :

```bash
pip install opencv-python ultralytics python-osc
```

Note : Lors de la première exécution, le script téléchargera automatiquement le fichier de poids du modèle (yolov8n-pose.pt).

## Utilisation
Assurez-vous que votre webcam est branchée.
Lancez le script :
python main.py
Le script détectera automatiquement votre matériel (CPU, NVIDIA CUDA, ou Apple MPS) pour optimiser les performances.
Appuyez sur la touche 'q' pour quitter.
Format des messages OSC
Le script envoie les données à l'adresse 127.0.0.1 sur le port 9000.

Adresse : /p{id_personne}/{partie_du_corps}
Arguments : float x, float y (valeurs normalisées entre 0.0 et 1.0)

Exemples :
* /p0/nose : Position du nez de la 1ère personne.
* /p0/right_wrist : Poignet droit de la 1ère personne.

Liste des points suivis (Keypoints COCO) :
nose, left_eye, right_eye, left_ear, right_ear, left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle.

