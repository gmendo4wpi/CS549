import tensorflow as tf
import cv2
import numpy as np

# Load trained model
model = tf.keras.models.load_model('mnist_digit_recognizer.keras')

# Initialize camera
cap = cv2.VideoCapture(0)  # Use 0 for default camera

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Define image size expected by model
img_height, img_width = 28, 28

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Failed to capture image.")
        break
    
    # Convert  frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Use thresholding to convert grayscale image to binary
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours in binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        # Get bounding box for each contour
        x, y, w, h = cv2.boundingRect(contour)
        
        # Ignore small contours that could be noise
        if w > 10 and h > 10:
            # Extract region of interest
            roi = binary[y:y+h, x:x+w]
            
            # Resize ROI to 28x28 pixels (size expected by model)
            roi_resized = cv2.resize(roi, (img_height, img_width))
            
            # Normalize pixel values
            roi_normalized = roi_resized / 255.0
            
            # Expand dimensions to match input shape of model
            roi_expanded = np.expand_dims(roi_normalized, axis=(0, -1))
            
            # Make predictions
            predictions = model.predict(roi_expanded)
            predicted_digit = np.argmax(predictions)
            
            # Draw bounding box and predicted digit on frame
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, str(predicted_digit), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Display resulting frame
    cv2.imshow('Digit Recognizer', frame)
    
    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release capture
cap.release()
cv2.destroyAllWindows()
