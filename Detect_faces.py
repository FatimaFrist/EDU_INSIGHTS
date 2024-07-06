import os
import cv2
import dlib 


def detect_faces(input_folder, output_folder):
    # Vérifier si le dossier de sortie existe, sinon le créer
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Initialiser le détecteur de visages de dlib
    # Ce détecteur utilise un modèle de machine learning pour identifier 
    # les caractéristiques typiques d'un visage humain et localiser ces visages dans une image.
    face_detector = dlib.get_frontal_face_detector()

    # Liste des fichiers dans le dossier d'entrée
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

    for image_file in image_files:
        # Chemin complet de l'image d'entrée
        input_path = os.path.join(input_folder, image_file)

        # Lire l'image
        image = cv2.imread(input_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Détecter les visages dans l'image
        faces = face_detector(gray_image)

        # Vérifier si des visages ont été détectés
        if faces:
            # Dessiner des rectangles autour des visages sur l'image
            image_with_faces = image.copy()
            for face in faces:
                cv2.rectangle(image_with_faces, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)

            # Chemin complet de l'image de sortie avec les visages détectés
            output_path_with_faces = os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}.jpg")
            cv2.imwrite(output_path_with_faces, image_with_faces)
    print(f"les Visages détectés sur les images sont bien enregistrée dans ce dossier : {output_folder}")
