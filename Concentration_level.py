import os
import cv2
import dlib
import numpy as np

def detect_concentration(input_folder, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Initialize Dlib's face detector and facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # Iterate through all images in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Detect faces in the image
            faces = detector(gray)

            for face in faces:
                # Predict facial landmarks
                landmarks = predictor(gray, face)

                # Extract coordinates for left and right eyes
                left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
                right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]

                # Calculate the aspect ratio of eyes
                left_eye_aspect_ratio = eye_aspect_ratio(left_eye)
                right_eye_aspect_ratio = eye_aspect_ratio(right_eye)

                # Average the aspect ratios of both eyes
                avg_aspect_ratio = (left_eye_aspect_ratio + right_eye_aspect_ratio) / 2.0

                # Define threshold for concentration
                concentration_threshold = 0.30

                # Determine concentration level based on threshold
                if avg_aspect_ratio >= concentration_threshold:
                    concentration_level = "High"
                    color = (0, 255, 0)  # Green for high concentration
                else:
                    concentration_level = "Low"
                    color = (0, 0, 255)  # Red for low concentration

                # Draw rectangle around the face
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)

                # Annotate image with concentration level
                cv2.putText(image, f'Concentration: {concentration_level}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Save the processed image to the output folder
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, image)

def eye_aspect_ratio(eye):
    # Compute the euclidean distances between the two sets of vertical eye landmarks
    A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
    B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))

    # Compute the euclidean distance between the horizontal eye landmark
    C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))

    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    return ear


