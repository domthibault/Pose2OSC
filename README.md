# Suivi de Pose OSC

Ce projet capture les mouvements du corps humain via webcam et les envoie via OSC pour un usage musical ou visuel.

## Étape 3 : Communication OSC

Cette version envoie les coordonnées normalisées (0.0 à 1.0) des points du corps via le protocole OSC.

### Prérequis

*   Python 3.8+
*   Webcam

### Installation

Installez les dépendances (ajout de `python-osc`) :

```bash
pip install opencv-python ultralytics python-osc