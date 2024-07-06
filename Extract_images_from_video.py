import cv2
import os

def extract_images(video_path, output_folder, interval_between_images=100):
    # Extraire le nom de la vidéo sans extension
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # Ouvrir la vidéo
    cap = cv2.VideoCapture(video_path)

    # Variables pour le nombre total d'images extraites
    total_images_extracted = 0

    # Créer le dossier principal de sortie s'il n'existe pas déjà
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Boucle de lecture de la vidéo
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Vérifier si c'est le moment d'extraire une image
        if total_images_extracted % interval_between_images == 0:
            # Enregistrer le frame comme une image dans le dossier de sortie principal
            image_name = os.path.join(output_folder, f"{video_name}_frame_{total_images_extracted}.jpg")
            cv2.imwrite(image_name, frame)

        total_images_extracted += 1

    # Libérer la vidéo
    cap.release()

    print("Les images ont été extraites avec succès.")


