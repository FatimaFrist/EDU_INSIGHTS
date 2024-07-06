# Core Packages
import streamlit as st
import cv2
from PIL import Image, ImageEnhance
import numpy as np
import os
import tempfile
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import Extract_images_from_video as extract
import random
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from csv_processing import *



face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# face_detector = dlib.get_frontal_face_detector()

try:
    emotion_model = load_model('fer2013_mini_XCEPTION.102-0.66.hdf5', compile=False)
except Exception as e:
    st.error(f"Error loading emotion detection model: {e}")


# Définir les émotions
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

def plot_emotion_pie_chart(emotion_data):
    labels = emotion_data.keys()
    sizes = emotion_data.values()
    
    fig, ax = plt.subplots()
    wedges, texts, autotexts = ax.pie(sizes, autopct='%1.1f%%', startangle=140)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    
    # Customize legend
    ax.legend(wedges, labels,
              title="Emotions",
              loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1))
    
    st.pyplot(fig)

# Function to plot pie chart for behavior distribution
def plot_behavior_pie_chart(behavior_data):
    labels = list(behavior_data.keys())
    sizes = list(behavior_data.values())

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    st.pyplot(fig)
    
def detect_faces(our_image):
    try:
        new_img = np.array(our_image.convert('RGB'))
        img = cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Use the BGR image for conversion
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert back to RGB for display
        return img, faces 
    except Exception as e:
        st.error(f"Error detecting faces: {e}")
        return None, None

def detect_emotions(our_image):
    new_img = np.array(our_image.convert('RGB'))
    img = cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Use the BGR image for conversion
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
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # Afficher l'émotion détectée
        cv2.putText(img, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert back to RGB for display
    return img



def detect_faces_in_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return frame, faces



def detect_behaviors(our_image):
    try:
        # Convert PIL image to numpy array and BGR format
        img = np.array(our_image.convert('RGB'))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        print("Image converted successfully.")

        # Convert image to grayscale
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Apply binary thresholding to extract potential areas of different activities
        _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

        # Find contours of potential activities
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        print(f"Number of contours found: {len(contours)}")

        # Placeholder for detected behaviors
        behaviors_detected = []

        # Define minimum area to consider a contour as a behavior
        min_area = 1000

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                # Simple aspect ratio-based behavior detection
                if w > h:
                    behaviors_detected.append("Writing")
                else:
                    behaviors_detected.append("Reading")

        print(f"Behaviors detected: {behaviors_detected}")

        # Default behavior if none detected
        if not behaviors_detected:
            behaviors_detected.append("Standing")

        # Select the first detected behavior (you can customize this logic)
        behavior_label = behaviors_detected[0]

        # Draw behavior label on the image
        cv2.putText(img_bgr, f" {behavior_label}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Convert back to RGB for display
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        print("Image processed successfully.")

        return img_rgb

    except Exception as e:
        print(f"Error during processing: {e}")
        return None

def upload_video():
    uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        video_name_with_extension = uploaded_file.name  # Save the name of the uploaded file
        video_name, extension = os.path.splitext(video_name_with_extension)  # Get the name without extension
        st.video(uploaded_file)
        
        # Save the uploaded video to the Videos directory
        video_dir = "Videos"
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)
        
        video_path = os.path.join(video_dir, video_name_with_extension)
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        return uploaded_file, video_name, video_name_with_extension, video_path
    else:
        return None, None, None, None

def main():
    """E-learning Insights """
    st.title("E-learning Insights")

   # activities = ["Process Images", "Process Videos", "Process CSV files"]
    #choice = st.sidebar.selectbox("Select Activity", activities)

    # Define activities as a dictionary with page content
    activities = {
        "Process Images": "Page for processing images.",
        "Process Videos": "Page for processing videos.",
        "Process CSV files": "Page for processing CSV files."
    }

    # Sidebar to select activity
    choice = st.sidebar.radio("Choose an activity", list(activities.keys()))

    if choice == 'Process Images':
        st.subheader("Process Images")

        image_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])

        if image_file is not None:
            our_image = Image.open(image_file)
            st.text("Original Image")
            st.image(our_image)

            task = ["Faces", "Emotions","Behaviors"]    
            feature_choice = st.selectbox("Find Features", task)
            if st.button("Process"):
                if feature_choice == 'Faces':
                    result_img, result_faces = detect_faces(our_image)
                    st.image(result_img)
                    st.success(f"Found {len(result_faces)} faces")
                elif feature_choice == 'Emotions':
                    result_img = detect_emotions(our_image)
                    st.image(result_img)
                elif feature_choice == 'Behaviors':
                    result_img = detect_behaviors(our_image)
                    st.image(result_img)

    elif choice == 'Process Videos':
        st.subheader("Process Videos")
        # Initialize session state if not already done

        if 'process_running' not in st.session_state:
            st.session_state.process_running = False
            st.session_state.images_extracted = False
        
        uploaded_file, video_name, video_name_with_extension, video_path = upload_video()

        if uploaded_file is not None and video_name_with_extension is not None:
            st.write(f"Selected video name: {video_name_with_extension}")
            output_extracted = f"Extracted_images/{video_name}"  # Output directory
            output_emotions = f"Emotions_detected/{video_name}"
            output_emotions_vd = "Emotions_videos"
            output_data_folder = 'Data_emotions'
            output_concentraions = f"Level_concentration/{video_name}"
            output_concentraions_vd = f"Concentration_videos"
            output_data_folder_concentration = 'Data_concentration'
            output_behaviors_vd = "Behaviors_videos"


            status_placeholder = st.empty()

            # Check if images have been extracted already
            if not st.session_state.images_extracted:
                status_placeholder.write(":hourglass_flowing_sand: Please wait while we process your video...")
                extract.extract_images(video_path, output_extracted)
                st.session_state.images_extracted = True
                status_placeholder.empty()

            if st.button("Detect Emotions !"):
                # Create a placeholder for the status message
                status_placeholder.write(":hourglass_flowing_sand: Please wait while we Detect Emotions of your Students ...")
                # Processing the video
                import Detect_emotions as emotions
                import detect_emotions_vd as emotions_vd
                emotions.process_images(output_extracted, output_emotions)
                #emotions_vd.analyze_video_emotions(video_path, output_emotions_vd, output_data_folder)
                
                status_placeholder.empty()

                # Show images of detected emotions in a 3x3 grid
                st.title("Detected Emotions")
                image_files = os.listdir(output_emotions)

                # Select 9 random images
                random.shuffle(image_files)
                selected_images = image_files[:9]

                # Calculate the number of columns and rows
                num_cols = 3
                num_rows = 3

                # Display images in a 3x3 grid
                columns = st.columns(num_cols)
                for i in range(len(selected_images)):
                    columns[i % num_cols].image(f"Emotions_detected/{video_name}/{selected_images[i]}", caption=selected_images[i], use_column_width=True)

                # Show resulting video
                st.title("Resulting Video")
                video_file_path = os.path.join(output_emotions_vd, f'{video_name}.mp4')
        

                if os.path.exists(video_file_path):
                    st.video(video_file_path)
                else:
                    st.error(f"Video file not found: {video_file_path}")

                # Read emotion data from text file and display pie chart
                emotion_data = {}
                emotion_file_path = f"{output_data_folder}/{video_name}.txt"
                if os.path.exists(emotion_file_path):
                    with open(emotion_file_path, 'r') as file:
                        for line in file:
                            emotion, percentage = line.strip().split(': ')
                            emotion_data[emotion] = float(percentage.strip('%'))

                    st.title("Emotion Distribution")
                    plot_emotion_pie_chart(emotion_data)

                else:
                    st.error(f"Emotion data file not found: {emotion_file_path}")

            if st.button("Detect Level of concentration !"): 
                # Create a placeholder for the status message
                status_placeholder = st.empty()
                status_placeholder.write(":hourglass_flowing_sand: Please wait while we Detect concentration's level of your Students ...")

                # Processing the video
                import Concentration_level as concentraion
                import Concentration_level_vd as concentration_vd

                concentraion.detect_concentration(output_extracted, output_concentraions)
                #concentration_vd.accelerate_video_with_concentration(video_path, output_concentraions_vd, output_data_folder_concentration)

                # Clear the status message
                status_placeholder.empty()

                # Show images of detected concentration levels in a 3x3 grid
                st.title("Detected Concentration's Level")
                image_files = os.listdir(output_concentraions)

                # Select 9 random images
                random.shuffle(image_files)
                selected_images = image_files[:9]

                # Calculate the number of columns and rows
                num_cols = 3
                num_rows = 3

                # Display images in a 3x3 grid
                columns = st.columns(num_cols)
                for i in range(len(selected_images)):
                    columns[i % num_cols].image(f"Level_concentration/{video_name}/{selected_images[i]}", caption=selected_images[i], use_column_width=True)

                # Show resulting video
                st.title("Resulting Video of Concentration's level")
                video_file_path = f"{output_concentraions_vd}/{video_name}.mp4"
                print(video_file_path)
                st.video(video_file_path)

                # Read concentration level counts from the text file
                concentration_file_path = os.path.join(output_data_folder_concentration, f"{video_name}.txt")
                with open(concentration_file_path, 'r') as file:
                    lines = file.readlines()
                    low_concentration_count = int(lines[0].split(': ')[1])
                    high_concentration_count = int(lines[1].split(': ')[1])

                # Calculate the total frames
                total_frames = low_concentration_count + high_concentration_count

                # Calculate the frequency of low and high concentration frames
                low_concentration_freq = low_concentration_count / total_frames * 100  # Convert to percentage
                high_concentration_freq = high_concentration_count / total_frames * 100  # Convert to percentage

                # Create a doughnut chart for the concentration levels
                st.title("Concentration Level Frequencies")
                fig, ax = plt.subplots()
                wedges, texts, autotexts = ax.pie(
                    [low_concentration_freq, high_concentration_freq],
                    # labels=['Low Concentration', 'High Concentration'],
                    colors=['red', 'green'],
                    autopct='%1.1f%%',
                    startangle=90,
                    wedgeprops=dict(width=0.3)
                )
                ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
                plt.setp(autotexts, size=10, weight="bold")
                plt.legend(wedges, ['Low Concentration', 'High Concentration'], loc="best")
                st.pyplot(fig)
        
    elif choice == "Process CSV files":
        st.subheader("Process CSV files")

        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            df = upload_and_display_csv(uploaded_file)
            if df is not None:
                st.write(df)
                selected_columns = st.multiselect('Select columns for analysis', df.columns.tolist())
                if selected_columns:
                    df_selected = select_columns(df, selected_columns)
                    if df_selected is not None:
                        st.write(df_selected)

                        operation = "Clustering"

                        if operation == "Clustering":
                            df_scaled, encoder, scaler, categorical_columns = preprocess_data(df_selected)
                            optimal_clusters, silhouette_scores = determine_optimal_clusters(df_scaled)
                            st.write(f"Optimal number of clusters: {optimal_clusters}")

                            # Slider for user to select number of clusters
                            n_clusters = st.slider("Select number of clusters", 2, 10, optimal_clusters)
                            kmeans, df_selected, cluster_means = perform_clustering(df_selected, df_scaled, n_clusters)

                            # Visualize clusters with a pie chart
                            cluster_counts = df_selected['cluster'].value_counts()
                            fig, ax = plt.subplots()
                            wedges, texts, autotexts = ax.pie(cluster_counts, autopct='%1.1f%%', startangle=90)
                            ax.legend(wedges, cluster_counts.index, title='Cluster', loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
                            ax.axis('equal')
                            plt.title('Cluster Distribution')
                            st.pyplot(fig)

                    

                            st.write("Clustered Data:")
                            st.write(df_selected)    

            
if __name__ == '__main__':
    main()
