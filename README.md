# Emotion-Based Music Recommendation System by Sloveshan Dayalan

## Overview

This project provides a web application that recommends songs based on the user's emotion, detected via facial and hand landmarks using a webcam. The system uses MediaPipe for emotion recognition and a neural network model for emotion classification.

## Prerequisites

- Python 3.x
- Required Python packages:
  - `streamlit`
  - `opencv-python`
  - `numpy`
  - `mediapipe`
  - `tensorflow` (for Keras)
  - `keras`

You can install the required packages using pip:

```bash
pip install streamlit opencv-python numpy mediapipe tensorflow keras
```

## Project Structure

- `collection.py`: Captures landmark data from the webcam and saves it as .npy files.
- `training.py`: Trains a neural network model using the collected data and saves the trained model and labels.
- `main.py`: Provides a Streamlit web application for emotion-based music recommendation.
- `emotion.npy`: Stores the current detected emotion.

## Running the Project

### Collect Data

- Run `python data_collection.py` to collect facial and hand landmark data. This script will prompt you for a name to save the collected data/emotion

### Train the Model

- Once you have collected enough data, run `python data_training.py` to train the model:
- This script will process the collected data, train the model, and save it as model.h5. It will also save the label mappings in labels.npy.

### Run the Web Application

- After training the model, you can run the Streamlit web application with `streamlit run main.py`.

### Using the Web Application

- Select Language: Choose your preferred language from the dropdown.
- Select Singer: Choose a singer from the dropdown based on the selected language.
- Capture Emotion: Click the "Capture Emotion" button to capture your emotion using the webcam.
- Recommend Songs: After capturing your emotion, click the "Recommend me songs" button to open YouTube with recommendations based on your mood.

## How It Works

- `data_collection.py` captures facial and hand landmarks from the webcam and saves them as .npy files in the ./emotions folder.
  It processes the frames using MediaPipe's Holistic model to extract landmarks.
  Model Training:

- `data_training.py` loads the collected data, prepares it for training, and trains a neural network to classify emotions based on the landmarks.
  The trained model and label mappings are saved as model.h5 and labels.npy, respectively.
  Web Application:

- `main.py` uses Streamlit to create a user interface for capturing emotions and recommending songs.
  The application loads the trained model and label mappings, captures emotions in real-time, and recommends songs based on the detected emotion.

## Troubleshooting

- Webcam Issues: Ensure that your webcam is properly connected and accessible.
- Data Collection Errors: Make sure to capture enough data for each emotion. The script stops after 100 frames by default.
- Model Training Issues: Ensure that the collected data is correctly formatted and sufficient for training.

## Sloveshan Dayalan (st20208680) (CL/BSCSD/27/78)
