import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load saved model
model_path = "C:/Users/Gio/Desktop/tinyml/archive/flower_classifier_model_v2.h5"
model = tf.keras.models.load_model(model_path)

# Define class names
class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# Initialize camera
cap = cv2.VideoCapture(0)  # Use 0 for default camera

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Define image size expected by model
img_height, img_width = 224, 224

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Failed to capture image.")
        break
    
    # Preprocess frame
    img = cv2.resize(frame, (img_height, img_width))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Make predictions
    predictions = model.predict(img)
    predicted_class = class_names[np.argmax(predictions)]

    # Display resulting frame
    cv2.putText(frame, predicted_class, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Flower Classifier', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release capture
cap.release()
cv2.destroyAllWindows()
