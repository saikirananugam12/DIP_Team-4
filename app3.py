# Install required packages
!pip install gradio deepface opencv-python pandas openpyxl

import gradio as gr
from deepface import DeepFace
import cv2
import pandas as pd
import pickle

# Global variable to hold the loaded model
loaded_model = None

# Function to load the .pkl file
def load_model(pkl_file):
    global loaded_model
    try:
        with open(pkl_file.name, 'rb') as f:
            loaded_model = pickle.load(f)
        return {"Status": "Model loaded successfully"}
    except Exception as e:
        return {"Error": str(e)}

# Function to categorize age into age groups
def categorize_age(age):
    if age <= 2:
        return "0-2"
    elif age <= 9:
        return "3-9"
    elif age <= 19:
        return "10-19"
    elif age <= 29:
        return "20-29"
    elif age <= 39:
        return "30-39"
    elif age <= 49:
        return "40-49"
    elif age <= 59:
        return "50-59"
    elif age <= 69:
        return "60-69"
    else:
        return "70+"

# Function to analyze a single image and extract attributes
def detect_emotion(image):
    global loaded_model
    if loaded_model is None:
        return {"Error": "Please upload a .pkl file first"}

    try:
        # Analyze the image using DeepFace
        emotion = DeepFace.analyze(image, actions=['age', 'gender', 'race'])

        # Extract key details
        age = emotion[0]['age']
        dominant_gender = emotion[0]['dominant_gender']
        dominant_race = emotion[0]['dominant_race']

        # Map gender and categorize age
        mapped_gender = "male" if dominant_gender.lower() == "man" else "female"
        age_group = categorize_age(age)

        # Format the result as a dictionary
        result = {
            "Age Group": age_group,
            "Gender": mapped_gender,
            "Race": dominant_race,
        }
        return result
    except Exception as e:
        return {"Error": str(e)}

# Function to analyze video and extract attributes from multiple frames
def detect_emotion_from_video(video):
    global loaded_model
    if loaded_model is None:
        return {"Error": "Please upload a .pkl file first"}

    try:
        # Load the video using OpenCV
        cap = cv2.VideoCapture(video)
        results = []
        frame_count = 0

        # List to store results for saving to Excel
        results_list = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process every nth frame to reduce computation (adjust n as needed)
            frame_count += 1
            if frame_count % 30 == 0:  # Analyze every 30th frame
                try:
                    # Convert the frame from BGR to RGB (OpenCV uses BGR)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Analyze the frame with DeepFace
                    emotions = DeepFace.analyze(frame_rgb, actions=['age', 'gender', 'race'], enforce_detection=False)

                    # If only one person is detected, ensure it's in list format
                    if isinstance(emotions, dict):
                        emotions = [emotions]

                    # Process each detected face
                    for emotion in emotions:
                        age = emotion.get('age', None)
                        dominant_gender = emotion.get('dominant_gender', None)
                        dominant_race = emotion.get('dominant_race', None)

                        # Map gender and categorize age
                        mapped_gender = "male" if dominant_gender and dominant_gender.lower() == "man" else "female"
                        age_group = categorize_age(age) if age is not None else "Unknown"

                        # Save the result
                        result = {
                            "Frame": frame_count,
                            "Age Group": age_group,
                            "Gender": mapped_gender,
                            "Race": dominant_race,
                        }
                        results.append(result)
                        results_list.append(result)

                except Exception as e:
                    print(f"Error processing frame {frame_count}: {e}")

        cap.release()
        # Save results to Excel
        df = pd.DataFrame(results_list)
        output_file = "face_attributes_results.xlsx"
        df.to_excel(output_file, index=False)

        # Return results for JSON output
        return results
    except Exception as e:
        return {"Error": str(e)}

# Unified Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Face Attribute Detection with Custom Model")
    gr.Markdown("Upload a `.pkl` file to load your model, then analyze an image or video.")

    # Tab for uploading .pkl file
    with gr.Tab("Upload Model"):
        pkl_input = gr.File(label="Upload .pkl File")
        pkl_output = gr.JSON(label="Model Load Status")
        pkl_btn = gr.Button("Load Model")
        pkl_btn.click(fn=load_model, inputs=pkl_input, outputs=pkl_output)

    # Tab for image input
    with gr.Tab("Image Input"):
        image_input = gr.Image(type="numpy", label="Upload an Image")
        image_output = gr.JSON(label="Detected Attributes from Image")
        image_btn = gr.Button("Analyze Image")
        image_btn.click(fn=detect_emotion, inputs=image_input, outputs=image_output)

    # Tab for video input
    with gr.Tab("Video Input"):
        video_input = gr.Video(label="Upload a Video")
        video_output = gr.JSON(label="Detected Attributes from Video")
        video_btn = gr.Button("Analyze Video")
        video_btn.click(fn=detect_emotion_from_video, inputs=video_input, outputs=video_output)

# Launch the Gradio app in public mode to share a link
demo.launch(share=True)

