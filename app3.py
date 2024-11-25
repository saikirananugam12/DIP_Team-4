import gradio as gr
from deepface import DeepFace
import cv2
import pandas as pd

# Function to analyze a single image and extract attributes
def detect_emotion(image):
    try:
        # Analyze the image using DeepFace
        emotion = DeepFace.analyze(image, actions=['age', 'gender', 'race'])
        
        # Extract key details
        age = emotion[0]['age']
        dominant_gender = emotion[0]['dominant_gender']
        dominant_race = emotion[0]['dominant_race']
        # dominant_emotion = emotion[0]['dominant_emotion']
        
        # Format the result as a dictionary
        result = {
            "Age": age,
            "Gender": dominant_gender,
            "Race": dominant_race,
            # "Emotion": dominant_emotion
        }
        return result
    except Exception as e:
        return {"Error": str(e)}

# Function to analyze video and extract attributes from multiple frames
def detect_emotion_from_video(video):
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
            if frame_count % 30 == 0:  # Analyze every 2nd frame
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
                        # dominant_emotion = emotion.get('dominant_emotion', None)

                        # Save the result
                        result = {
                            "Frame": frame_count,
                            "Age": age,
                            "Gender": dominant_gender,
                            "Race": dominant_race,
                            # "Emotion": dominant_emotion
                        }
                        results.append(result)
                        results_list.append(result)

                except Exception as e:
                    print(f"Error processing frame {frame_count}: {e}")

        cap.release()

        # Save results to Excel
        df = pd.DataFrame(results_list)
        output_file = "face_attributes_results_2.xlsx"
        df.to_excel(output_file, index=False)

        # Return results for JSON output
        return results
    except Exception as e:
        return {"Error": str(e)}

# Unified Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("Group-4")

    gr.Markdown("# Face Attribute Detection")
    gr.Markdown("Upload an image or a video to detect age, gender,race,from faces.")
    
    with gr.Tab("Image Input"):
        image_input = gr.Image(type="numpy", label="Upload an Image")
        image_output = gr.JSON(label="Detected Attributes from Image")
        image_btn = gr.Button("Analyze Image")
        image_btn.click(fn=detect_emotion, inputs=image_input, outputs=image_output)
    
    with gr.Tab("Video Input"):
        video_input = gr.Video(label="Upload a Video")
        video_output = gr.JSON(label="Detected Attributes from Video")
        video_btn = gr.Button("Analyze Video")
        video_btn.click(fn=detect_emotion_from_video, inputs=video_input, outputs=video_output)

# Launch the Gradio app
demo.launch(share=True)
