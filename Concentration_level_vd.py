import os
import cv2
import dlib
import numpy as np

def eye_aspect_ratio(eye):
    # Compute the euclidean distances between the two sets of vertical eye landmarks
    A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
    B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))

    # Compute the euclidean distance between the horizontal eye landmark
    C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))

    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    return ear

def detect_concentration(frame, detector, predictor):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = detector(gray)

    # Initialize variables for concentration level detection
    concentration_level = "Low"
    color = (0, 0, 255)  # Red for low concentration

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

    return concentration_level, color

def accelerate_video_with_concentration(input_video, output_folder, output_data_folder, frame_skip=2, consecutive_frames_threshold=4):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    # Create output data folder if it doesn't exist
    if not os.path.exists(output_data_folder):
        os.makedirs(output_data_folder)

    # Initialize Dlib's face detector and facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # Open the video file
    cap = cv2.VideoCapture(input_video)

    # Get the frame width and height
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_name = os.path.splitext(os.path.basename(input_video))[0]

    # Define the codec and create VideoWriter object
    out = cv2.VideoWriter(os.path.join(output_folder, f'{video_name}.mp4'), cv2.VideoWriter_fourcc(*'avc1'), 30, (frame_width, frame_height))

    # Initialize variables for tracking concentration level
    consecutive_frames = 0
    current_concentration_level = "Low"
    current_concentration_color = (0, 0, 255)  # Red for low concentration

    # Initialize counters for concentration levels
    low_concentration_count = 0
    high_concentration_count = 0

    # Read until video is completed
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Skip frames
            for _ in range(frame_skip - 1):
                cap.read()

            # Detect concentration level
            concentration_level, color = detect_concentration(frame, detector, predictor)

            # Update current concentration level
            if concentration_level == current_concentration_level:
                consecutive_frames += 1
            else:
                consecutive_frames = 0
                current_concentration_level = concentration_level
                current_concentration_color = color

            # If concentration level remains the same for consecutive frames, annotate the frame
            if consecutive_frames >= consecutive_frames_threshold:
                # Detect concentration level again for accurate face position
                concentration_level, color = detect_concentration(frame, detector, predictor)
                # Draw rectangle around the face
                faces = detector(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                for face in faces:
                    x, y, w, h = face.left(), face.top(), face.width(), face.height()
                    cv2.rectangle(frame, (x, y), (x+w, y+h), current_concentration_color, 2)
                    # Annotate frame with concentration level
                    cv2.putText(frame, f'Concentration: {current_concentration_level}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, current_concentration_color, 2)

                # Update concentration level counters
                if current_concentration_level == "Low":
                    low_concentration_count += 1
                else:
                    high_concentration_count += 1

            # Write the frame into the output video
            out.write(frame)

        else:
            break

    # Release the video capture and video writer objects
    cap.release()
    out.release()

    # Write concentration level counts to a text file
    with open(os.path.join(output_data_folder, f'{video_name}.txt'), 'w') as f:
        f.write(f'Low concentration frames: {low_concentration_count}\n')
        f.write(f'High concentration frames: {high_concentration_count}\n')

