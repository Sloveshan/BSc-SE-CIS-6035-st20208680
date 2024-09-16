import os  
import numpy as np 
import cv2 
from tensorflow.keras.utils import to_categorical
from keras.layers import Input, Dense 
from keras.models import Model

is_init = False
size = -1

label = []
dictionary = {}
c = 0

X_list = []
y_list = []

# Iterate over all .npy files in the directory
for i in os.listdir('./emotions'):  # Fetch files from the './emotions' folder
    if i.endswith(".npy") and not i.startswith(("labels", "emotion")):  
        data = np.load(os.path.join('./emotions', i))  # Load .npy file
        
        if not is_init:
            size = data.shape[0]
            y = np.array([i.split('.')[0]] * size).reshape(-1, 1)
            is_init = True
        else:
            # Check dimension
            if data.ndim != X_list[0].ndim:
                print(f"Error: Dimensions mismatch. Current file dimensions: {data.ndim}, First file dimensions: {X_list[0].ndim}")
                continue 
            
            size = data.shape[0]
            y = np.array([i.split('.')[0]] * size).reshape(-1, 1)
        
        X_list.append(data)
        y_list.append(y)
        
        label.append(i.split('.')[0])
        dictionary[i.split('.')[0]] = c  
        c += 1

# Convert lists to numpy arrays
X = np.concatenate(X_list, axis=0)
y = np.concatenate(y_list, axis=0)

# Map labels to integers
for i in range(y.shape[0]):
    y[i, 0] = dictionary[y[i, 0]]
y = np.array(y, dtype="int32")

# Convert labels to categorical format
y = to_categorical(y)

# Shuffle data
X_new = X.copy()
y_new = y.copy()
counter = 0 
cnt = np.arange(X.shape[0])
np.random.shuffle(cnt)

for i in cnt: 
    X_new[counter] = X[i]
    y_new[counter] = y[i]
    counter += 1

# Define and compile the model
ip = Input(shape=(X.shape[1],))
m = Dense(512, activation="relu")(ip)
m = Dense(256, activation="relu")(m)
op = Dense(y.shape[1], activation="softmax")(m) 
model = Model(inputs=ip, outputs=op)
model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=['acc'])

# Train the model
model.fit(X_new, y_new, epochs=50)

# Save the model and labels
model.save("model.h5")
np.save("labels.npy", np.array(label))
