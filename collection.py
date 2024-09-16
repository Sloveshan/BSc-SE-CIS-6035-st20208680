import os
import mediapipe as mp
import numpy as np
import cv2

# Initialize video capture from the default camera
cap = cv2.VideoCapture(0)

# Prompt user for the name of the data file
name = input("Enter the name of the data : ")

# Define the folder to save the data
folder = './emotions'
# Create the folder if it doesn't exist
if not os.path.exists(folder):
    os.makedirs(folder)

# Initialize MediaPipe solutions for holistic landmarks (face, hands, pose)
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()  # Create an instance of the Holistic model
drawing = mp.solutions.drawing_utils  # Utilities for drawing landmarks

X = []  # List to store collected landmark data
data_size = 0  # Counter for the number of frames processed

while True:
    lst = []  # List to store landmarks for the current frame

    # Capture a frame from the video feed
    _, frm = cap.read()

    # Flip the frame horizontally for a mirror effect
    frm = cv2.flip(frm, 1)

    # Process the frame to extract landmarks
    res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

    # Check if face landmarks are detected
    if res.face_landmarks:
        # Extract face landmarks relative to the second landmark (nose tip)
        for i in res.face_landmarks.landmark:
            lst.append(i.x - res.face_landmarks.landmark[1].x)
            lst.append(i.y - res.face_landmarks.landmark[1].y)

        # Check if left hand landmarks are detected
        if res.left_hand_landmarks:
            # Extract left hand landmarks relative to the tip of the thumb
            for i in res.left_hand_landmarks.landmark:
                lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
        else:
            # Append zeros if left hand landmarks are not detected
            for i in range(42):
                lst.append(0.0)

        # Check if right hand landmarks are detected
        if res.right_hand_landmarks:
            # Extract right hand landmarks relative to the tip of the thumb
            for i in res.right_hand_landmarks.landmark:
                lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
        else:
            # Append zeros if right hand landmarks are not detected
            for i in range(42):
                lst.append(0.0)

        # Append the list of landmarks to the data array
        X.append(lst)
        data_size += 1  # Increment the frame counter

    # Draw landmarks on the frame
    drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_CONTOURS)
    drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
    drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

    # Display the number of frames processed on the frame
    cv2.putText(frm, str(data_size), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame in a window
    cv2.imshow("window", frm)

    # Break the loop if the 'ESC' key is pressed or 100 frames have been processed
    if cv2.waitKey(1) == 27 or data_size > 99:
        cv2.destroyAllWindows()  # Close all OpenCV windows
        cap.release()  # Release the video capture object
        break

# Save the collected landmark data to a NumPy file in the specified folder
file_path = os.path.join(folder, f"{name}.npy")
np.save(file_path, np.array(X))
# Print the shape of the saved data array
print(np.array(X).shape)
