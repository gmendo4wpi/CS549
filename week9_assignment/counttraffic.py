import cv2
import torch
import numpy as np
from google.colab import drive

# mount Google Drive
drive.mount('/content/drive')
import cv2

# Load YOLO model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def detect_objects(frame):
    """
    Function to detect objects in a frame using YOLOv5 model.
    """
    # Convert frame from BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame_rgb)
    return results.pandas().xyxy[0]

def is_inside_bounded_area(xmin, xmax, ymin, ymax, left_line_x, right_line_x, top_line_y, bottom_line_y):
    """
    Calculate percentage of bounding box inside bounded area and return it.
    """
    # Calculate area of bounding box
    bbox_area = (xmax - xmin) * (ymax - ymin)
    
    # Calculate overlap with bounded area
    overlap_xmin = max(xmin, left_line_x)
    overlap_xmax = min(xmax, right_line_x)
    overlap_ymin = max(ymin, top_line_y)
    overlap_ymax = min(ymax, bottom_line_y)
    
    # Calculate overlap area
    overlap_area = max(0, overlap_xmax - overlap_xmin) * max(0, overlap_ymax - overlap_ymin)
    
    # Calculate percentage of bounding box that is inside bounded area
    overlap_percentage = overlap_area / bbox_area
    
    return overlap_percentage

# Set up video capture
cap = cv2.VideoCapture('/content/drive/My Drive/TrafficVideo.mp4')

# Read first frame to get video properties
ret, frame = cap.read()
if not ret:
    print("Failed to load video.")
    cap.release()
    exit()

# Set up video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('/content/drive/My Drive/output6.avi', fourcc, 20.0, (640, 360))  # Adjusted to lower resolution output

frame_skip = 5  # Process every 5th frame
frame_count = 0

# Define coordinates for area of interest
x1, y1 = 415, 200
x2, y2 = 580, 210
x3, y3 = 520, 320
x4, y4 = 170, 285

# For simplicity, use average y-coordinate of blue lines
entry_line_y = (y1 + y2) // 2
exit_line_y = (y3 + y4) // 2

# Calculate x-coordinates for vertical magenta lines
left_magenta_line_x = min(x1, x4)
right_magenta_line_x = max(x2, x3)

# Initialize counters
person_count = 0.0  # Initialize bike counter as a float
bike_count = 0.0    # Initialize bike counter as a float
car_count = 0.0     # Initialize car counter as a float

# Initialize dictionaries to keep track of persons, bikes, and cars
person_states = {}
bike_states = {}

def is_crossing_line(y_bottom, line_y):
    """
    Check if bottom y-coordinate of bounding box crosses line.
    """
    return y_bottom >= line_y

def is_crossing_left_line(x_left, line_x):
    """
    Check if left x-coordinate of bounding box crosses vertical line.
    """
    return x_left <= line_x

# Process each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Only process every nth frame
    if frame_count % frame_skip == 0:
        # Resize frame to lower resolution
        frame = cv2.resize(frame, (640, 360))

        # Draw horizontal lines in blue
        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue color (entry line)
        cv2.line(frame, (x3, y3), (x4, y4), (255, 0, 0), 2)  # Blue color (exit line)

        # Draw vertical lines in magenta
        cv2.line(frame, (x2, y2), (x3, y3), (255, 0, 255), 2)
        cv2.line(frame, (x4, y4), (x1, y1), (255, 0, 255), 2)

        # Detect objects in frame
        detections = detect_objects(frame)

        # Process each detected person
        for index, detection in detections.iterrows():
            if detection['name'] == 'person' and detection['confidence'] > 0.5:
                xmin, ymin, xmax, ymax = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
                person_id = f"{xmin}-{ymin}-{xmax}-{ymax}"  # Simple unique ID based on bounding box (can be improved)

                # Check if person is crossing entry line with their feet (bottom of bounding box)
                if person_id not in person_states:
                    person_states[person_id] = {'entered': False, 'exited': False, 'entry_frame': 0, 'id': len(person_states) + 1}

                state = person_states[person_id]

                # Check for entry (crossing first blue line)
                if not state['entered'] and is_crossing_line(ymax, entry_line_y):
                    state['entered'] = True
                    state['entry_frame'] = frame_count  # Record frame number when person enters
                    cv2.putText(frame, f"Person {state['id']} entered", (xmin, ymin - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # Check for exit (crossing second blue line)
                if state['entered'] and not state['exited'] and is_crossing_line(ymax, exit_line_y):
                    state['exited'] = True
                    crossing_time = frame_count - state['entry_frame']  # Calculate number of frames it took to cross
                    person_count += 0.07  # Increment by 0.07 for each person that crosses both lines
                    cv2.putText(frame, f"Person {state['id']} crossed in {crossing_time} frames", (xmin, ymin - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                # Draw bounding box and label
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                label = f"Person {detection['confidence']:.2f}"
                cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            elif detection['name'] == 'bicycle' and detection['confidence'] > 0.5:
                xmin, ymin, xmax, ymax = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
                bike_id = f"{xmin}-{ymin}-{xmax}-{ymax}"  # Simple unique ID based on bounding box

                # Check if bike is crossing magenta lines (entry and exit)
                if bike_id not in bike_states:
                    bike_states[bike_id] = {'entered': False, 'exited': False, 'entry_frame': 0, 'id': len(bike_states) + 1}

                state = bike_states[bike_id]

                # Check for entry (crossing right magenta line)
                if not state['entered'] and is_crossing_left_line(xmin, right_magenta_line_x):
                    state['entered'] = True
                    state['entry_frame'] = frame_count  # Record frame number when bike enters
                    cv2.putText(frame, f"Bike {state['id']} entered", (xmin, ymin - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # Check for exit (crossing left magenta line)
                if state['entered'] and not state['exited'] and is_crossing_left_line(xmin, left_magenta_line_x):
                    state['exited'] = True
                    crossing_time = frame_count - state['entry_frame']  # Calculate number of frames it took to cross
                    bike_count += 0.20  # Increment by 0.20 for each bike that crosses both lines
                    cv2.putText(frame, f"Bike {state['id']} crossed in {crossing_time} frames", (xmin, ymin - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                # Draw bounding box and label for bikes
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2)
                label = f"Bike {detection['confidence']:.2f}"
                cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            elif detection['name'] == 'car' and detection['confidence'] > 0.5:
                xmin, ymin, xmax, ymax = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
                
                # Calculate percentage inside bounded area
                overlap_percentage = is_inside_bounded_area(xmin, xmax, ymin, ymax, left_magenta_line_x, right_magenta_line_x, entry_line_y, exit_line_y)
                
                # Check if car is inside bounded area and increment accordingly
                if 0.34 < overlap_percentage <= 0.38:
                    car_count += 0.25  # Increment by 0.25 if 34-38% inside
                    print(f"Car counted with 0.25 increment, overlap: {overlap_percentage*100:.2f}%")
                elif 0.39 < overlap_percentage <= 0.44:
                    car_count += 0.19  # Increment by 0.19 if 39-44% inside
                    print(f"Car counted with 0.19 increment, overlap: {overlap_percentage*100:.2f}%")
                elif 0.45 < overlap_percentage <= 0.55:
                    car_count += 0.18  # Increment by 0.18 if 45-55% inside
                    print(f"Car counted with 0.18 increment, overlap: {overlap_percentage*100:.2f}%")

                # Draw bounding box and label for cars
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 255, 0), 2)
                label = f"Car {detection['confidence']:.2f}"  # Display confidence level in white
                cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)  # White color

        # Update person, bike, and car count labels (rounded and displayed as integer)
        rounded_person_count = round(person_count)
        rounded_bike_count = round(bike_count)
        rounded_car_count = round(car_count)
        cv2.putText(frame, f'Persons: {rounded_person_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)  # White color
        cv2.putText(frame, f'Bikes: {rounded_bike_count}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)  # White color
        cv2.putText(frame, f'Cars: {rounded_car_count}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)  # White color

        # Write processed frame to output video
        out.write(frame)

    frame_count += 1

# Release everything when job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
print("Video processing complete. output has been saved.")
