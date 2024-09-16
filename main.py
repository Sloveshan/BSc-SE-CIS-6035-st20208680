import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import webbrowser

# Load model and labels
model = load_model("model.h5")
label = np.load("labels.npy")
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils


st.set_page_config(page_title="MoodTunes | D.Sloveshan", page_icon=":musical_note:", layout="wide")
st.title("Emotion-Based Music Recommender")
st.write("Select your language, choose a singer, and capture your emotion to get personalized song recommendations!")

# Initialize session state
if "run" not in st.session_state:
    st.session_state["run"] = True

try:
    emotion = np.load("emotion.npy")[0]
except:
    emotion = ""

if not emotion:
    st.session_state["run"] = True
else:
    st.session_state["run"] = False

# Function to capture emotion
def capture_emotion():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not open webcam.")
        return

    st.write("**Webcam is active. Look at the camera & press 'q' to capture your emotion.**")

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture image.")
            break

        # Flip the frame horizontally for a selfie display
        frame = cv2.flip(frame, 1)

        # Process the frame with MediaPipe for facial and hand landmarks
        res = holis.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Draw facial and hand landmarks on the frame
        if res.face_landmarks:
            drawing.draw_landmarks(frame, res.face_landmarks, holistic.FACEMESH_TESSELATION)
        if res.left_hand_landmarks:
            drawing.draw_landmarks(frame, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
        if res.right_hand_landmarks:
            drawing.draw_landmarks(frame, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

        # Show the live webcam feed in a new OpenCV window
        cv2.imshow("Webcam Feed - Press 'q' to capture emotion", frame)

        # Capture emotion landmarks when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            lst = []
            if res.face_landmarks:
                for i in res.face_landmarks.landmark:
                    lst.append(i.x - res.face_landmarks.landmark[1].x)
                    lst.append(i.y - res.face_landmarks.landmark[1].y)

                if res.left_hand_landmarks:
                    for i in res.left_hand_landmarks.landmark:
                        lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                        lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
                else:
                    for _ in range(42):
                        lst.append(0.0)

                if res.right_hand_landmarks:
                    for i in res.right_hand_landmarks.landmark:
                        lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                        lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
                else:
                    for _ in range(42):
                        lst.append(0.0)

                lst = np.array(lst).reshape(1, -1)

                # Predict emotion and save it
                pred = label[np.argmax(model.predict(lst))]
                np.save("emotion.npy", np.array([pred]))

                break  

    cap.release()
    cv2.destroyAllWindows()

    st.success(f"Emotion '{pred}' captured successfully!")

# Dictionary mapping languages to singers
singers_by_language = {
    "English": [
        "Ed Sheeran", "Travis Scott", "Adele", "Shawn Mendes", "Taylor Swift",
        "Ariana Grande", "Elton John", "Paul McCartney", "David Bowie", 
        "Freddie Mercury", "Amy Winehouse", "Sam Smith", "Dua Lipa",
        "Harry Styles", "Billie Eilish", "Bruno Mars", "Rihanna", "Coldplay"
    ],
    "Sinhala": [
        "Victor Ratnayake", "W.D. Amaradeva", "Nanda Malini", "Sunil Edirisinghe", 
        "Clarence Wijewardena", "Rohan Peiris", "Annesley Malewana", 
        "Kumar Sangakkara", "H.R. Jothipala", "Chandralekha Perera", 
        "Rukmani Devi", "Indrani Perera", "J. A. Milton Perera", "Sonia Dhanushka"
    ],
    "Tamil": [
        "Ilaiyaraaja", "A.R. Rahman", "K.J. Yesudas", "S.P. Balasubrahmanyam",
        "Chitra", "Hariharan", "Shankar Mahadevan", "K.S. Chithra",
        "Yuvan Shankar Raja", "Vijay Prakash", "Nithyasree Mahadevan",
        "G.V. Prakash Kumar", "Sunidhi Chauhan", "Anirudh Ravichander", "Srinivas"
    ],
    "Hindi": ["Arijit Singh", "Shreya Ghoshal", "Sonu Nigam", "Neha Kakkar"],
    "Korean": ["BTS", "BLACKPINK", "EXO"],
    "Spanish": ["Shakira", "Enrique Iglesias", "Bad Bunny"],
    "French": ["Édith Piaf", "Stromae", "Zaz"],
    "German": ["Rammstein", "Nena", "Helene Fischer"],
    "Japanese": ["Hikaru Utada", "Kenshi Yonezu", "LiSA"],
    "Chinese": ["Jay Chou", "G.E.M.", "Jackson Wang"]
}

col1, col2 = st.columns([2, 1])

with col1:
    # Dropdown for language selection
    lang = st.selectbox("Select a Language", list(singers_by_language.keys()))

    # Display singer dropdown based on selected language
    if lang:
        singer = st.selectbox(
            "Select a Singer",
            singers_by_language.get(lang, [])
        )

with col2:
    
    st.write("") 
    st.write("") 


    capture_btn = st.button("Capture Emotion")

    if capture_btn:
        capture_emotion()

    st.write("") 

    btn = st.button("Recommend me songs")

    if btn:
        if not (emotion):
            st.warning("Please capture your emotion first")
        else:
            webbrowser.open(f"https://www.youtube.com/results?search_query={emotion}+{lang}+song+by+{singer}")
            np.save("emotion.npy", np.array([""]))

with st.expander("About this App", expanded=True):
    st.write("""
        This application uses emotion detection from your webcam to recommend songs.
        It processes facial and hand landmarks to detect your emotion and then suggests songs that fit your mood.
    """)

