import cv2
import sys
import torch
from ultralytics import YOLO
from pythonosc.udp_client import SimpleUDPClient


def main():
    # --- CONFIGURATION OSC ---
    ip = "127.0.0.1"
    port = 9000
    client = SimpleUDPClient(ip, port)
    print(f"Client OSC initialisé sur {ip}:{port}")

    # Liste des noms des points clés (Standard COCO)
    keypoint_names = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]

    # --- DETECTION DU PERIPHERIQUE (GPU/CPU) ---
    # On vérifie si CUDA (NVIDIA) ou MPS (Mac Silicon) est dispo
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'

    print(f"--- Périphérique d'inférence sélectionné : {device.upper()} ---")

    # --- INITIALISATION CAMERA ---
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erreur : Impossible d'ouvrir la caméra.")
        sys.exit()

    # --- CHARGEMENT MODELE ---
    print("Chargement du modèle YOLOv8-pose...")
    # On charge le modèle. Note : on passera l'argument 'device' lors de l'inférence.
    model = YOLO('yolov8n-pose.pt')
    print("Modèle chargé.")
    print("Appuyez sur 'q' pour quitter.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- INFERENCE YOLO ---
        # On passe explicitement le device détecté (cuda, mps ou cpu)
        results = model(frame, device=device, verbose=False)

        # Visualisation
        annotated_frame = results[0].plot()
        cv2.imshow('Flux Camera - Etape 3 (OSC + GPU)', annotated_frame)

        # --- TRAITEMENT OSC ---
        if results[0].keypoints is not None and len(results[0].keypoints) > 0:

            # Récupération des données sur CPU pour l'envoi OSC
            # .cpu().numpy() est crucial ici car les tenseurs sont peut-être sur le GPU
            kpts_xyn = results[0].keypoints.xyn.cpu().numpy()
            kpts_conf = results[0].keypoints.conf.cpu().numpy()

            for i, (person_xyn, person_conf) in enumerate(zip(kpts_xyn, kpts_conf)):
                for kp_idx, ((x, y), conf) in enumerate(zip(person_xyn, person_conf)):
                    if conf > 0.5:
                        part_name = keypoint_names[kp_idx]
                        address = f"/p{i}/{part_name}"
                        client.send_message(address, [float(x), float(y)])

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Programme terminé.")


if __name__ == "__main__":
    main()