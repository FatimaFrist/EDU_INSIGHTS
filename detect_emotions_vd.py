import cv2
from tensorflow.keras.models import load_model
import numpy as np
import os

def analyze_video_emotions(video_path, output_video_folder, output_data_folder):
    # Charger le modèle de détection d'émotions (FER2013)
    emotion_model = load_model('fer2013_mini_XCEPTION.102-0.66.hdf5', compile=False)

    # Définir les émotions
    EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

    # Initialiser le classificateur de visages
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Fonction pour détecter les émotions sur une image
    def detect_emotions(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        emotions_count = {emotion: 0 for emotion in EMOTIONS}

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = image[y:y+h, x:x+w]

            # Vérifier si le visage est détecté
            if len(roi_gray) == 0 or len(roi_color) == 0:
                continue

            roi_gray = cv2.resize(roi_gray, (64, 64))  # Redimensionner l'image à (64, 64)
            roi_gray = roi_gray.astype("float") / 255.0
            roi_gray = np.expand_dims(roi_gray, axis=-1)  # Ajouter une dimension pour le canal de couleur
            roi_gray = np.expand_dims(roi_gray, axis=0)  # Ajouter une dimension pour le batch

            preds = emotion_model.predict(roi_gray)[0]
            emotion_label = EMOTIONS[preds.argmax()]

            emotions_count[emotion_label] += 1

            # Dessiner un rectangle autour du visage détecté
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # Afficher l'émotion détectée
            cv2.putText(image, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

        return image, emotions_count

    # Charger la vidéo
    cap = cv2.VideoCapture(video_path)

    # Récupérer les informations sur la vidéo (largeur, hauteur, fps)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Créer les dossiers pour enregistrer la vidéo modifiée et les fichiers de données d'émotion
    if not os.path.exists(output_video_folder):
        os.makedirs(output_video_folder)
    if not os.path.exists(output_data_folder):
        os.makedirs(output_data_folder)

    # Obtenir le nom de fichier de la vidéo d'origine
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # Créer un objet VideoWriter pour enregistrer la vidéo modifiée dans le dossier de sortie avec le même nom de fichier
    output_path = os.path.join(output_video_folder, f'{video_name}.mp4')
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (width, height))

    # Initialiser le dictionnaire pour compter les émotions
    total_emotions_count = {emotion: 0 for emotion in EMOTIONS}
    total_faces_detected = 0

    # Lire chaque image de la vidéo, détecter les émotions et enregistrer la vidéo avec les émotions détectées
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if frame_count % 10 == 0:  # Traiter chaque deuxième image pour accélérer le processus
                frame_with_emotions, emotions_count = detect_emotions(frame)
                out.write(frame_with_emotions)

                # Mettre à jour le nombre total d'émotions
                for emotion, count in emotions_count.items():
                    total_emotions_count[emotion] += count

                total_faces_detected += sum(emotions_count.values())

            frame_count += 1
        else:
            break

    # Libérer les ressources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Calculer les pourcentages d'émotions
    percentages_emotions = {emotion: (count / total_faces_detected) * 100 if total_faces_detected > 0 else 0 for emotion, count in total_emotions_count.items()}

    # Enregistrer les résultats dans un fichier texte
    data_path = os.path.join(output_data_folder, f'{video_name}.txt')
    with open(data_path, 'w') as f:
        for emotion, percentage in percentages_emotions.items():
            f.write(f"{emotion}: {percentage:.2f}%\n")

    print("Émotions détectées, vidéo enregistrée et fichier de données créé avec succès.")


