import cv2
from tensorflow.keras.models import load_model
import numpy as np
import os

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

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (64, 64))  # Redimensionner l'image à (64, 64)
        roi_gray = roi_gray.astype("float") / 255.0
        roi_gray = np.expand_dims(roi_gray, axis=-1)  # Ajouter une dimension pour le canal de couleur
        roi_gray = np.expand_dims(roi_gray, axis=0)  # Ajouter une dimension pour le batch

        preds = emotion_model.predict(roi_gray)[0]
        emotion_label = EMOTIONS[preds.argmax()]
        # Dessiner un rectangle autour du visage détecté
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # Afficher l'émotion détectée
        cv2.putText(image, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    return image

# Fonction principale pour traiter un dossier d'images
def process_images(input_folder, output_folder):
    # Vérifier si le dossier de sortie existe, sinon le créer
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Parcourir les images du dossier d'entrée
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Vous pouvez ajouter plus d'extensions si nécessaire
            image_path = os.path.join(input_folder, filename)
            # Lire l'image
            image = cv2.imread(image_path)
            # Détecter les émotions sur l'image
            image_with_emotions = detect_emotions(image)
            # Enregistrer l'image avec les émotions détectées dans le dossier de sortie
            cv2.imwrite(os.path.join(output_folder, filename), image_with_emotions)

    print("Émotions détectées et enregistrées avec succès dans le dossier de sortie.")

