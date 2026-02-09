import cv2
import sys
from ultralytics import YOLO
from pythonosc.udp_client import SimpleUDPClient


def main():
    # --- CONFIGURATION OSC ---
    # IP locale (localhost) et port standard.
    # Vous pourrez changer le port (ex: 8000, 9000, 5005) selon votre logiciel musical.
    ip = "127.0.0.1"
    port = 9000
    client = SimpleUDPClient(ip, port)
    print(f"Client OSC initialisé sur {ip}:{port}")

    # Liste des noms des points clés (Standard COCO utilisé par YOLO)
    keypoint_names = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]

    # --- INITIALISATION CAMERA ---
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erreur : Impossible d'ouvrir la caméra.")
        sys.exit()

    # --- CHARGEMENT MODELE ---
    print("Chargement du modèle YOLOv8-pose...")
    model = YOLO('yolov8n-pose.pt')
    print("Modèle chargé.")
    print("Appuyez sur 'q' pour quitter.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Inférence YOLO
        results = model(frame, verbose=False)

        # Visualisation (dessin des squelettes)
        annotated_frame = results[0].plot()
        cv2.imshow('Flux Camera - Etape 3 (OSC)', annotated_frame)

        # --- TRAITEMENT OSC ---
        # On vérifie si des squelettes ont été détectés
        if results[0].keypoints is not None and len(results[0].keypoints) > 0:

            # Récupération des données :
            # xyn : coordonnées normalisées (entre 0 et 1)
            # conf : score de confiance (entre 0 et 1)
            # On utilise .cpu().numpy() pour convertir les tenseurs PyTorch en tableaux utilisables
            kpts_xyn = results[0].keypoints.xyn.cpu().numpy()
            kpts_conf = results[0].keypoints.conf.cpu().numpy()

            # Boucle sur chaque personne détectée (i = index de la personne)
            for i, (person_xyn, person_conf) in enumerate(zip(kpts_xyn, kpts_conf)):

                # Boucle sur chaque point du corps de la personne
                for kp_idx, ((x, y), conf) in enumerate(zip(person_xyn, person_conf)):

                    # On n'envoie l'info que si la confiance est supérieure à 50%
                    if conf > 0.5:
                        part_name = keypoint_names[kp_idx]

                        # Construction de l'adresse OSC : /p0/nose, /p0/left_wrist, etc.
                        address = f"/p{i}/{part_name}"

                        # Envoi du message : adresse + arguments [x, y]
                        client.send_message(address, [float(x), float(y)])

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Programme terminé.")


if __name__ == "__main__":
    main()
