import cv2
import datetime
import numpy as np

def stitch_images(images):
    # create a Stitcher instance in panorama mode
    stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    # stitch images
    status, panorama = stitcher.stitch(images)
    if status == cv2.Stitcher_OK:
        return panorama
    else:
        print("Stitching couldn't be performed. Error code:", status)
        return None

# initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video device.")
    exit()

# set camera resolution
cap.set(3, 640)  # width
cap.set(4, 480)  # height

# function to update zoom level based on trackbar movement
def update_zoom(val):
    global zoom_level
    zoom_level = val
    
# function to update amount of blur based on trackbar movement
def update_blur(val):
    global blur_amount
    blur_amount = val

# create camera window & trackbars for zoom & blur
cv2.namedWindow('Camera')
cv2.createTrackbar('Zoom', 'Camera', 1, 10, update_zoom)
cv2.createTrackbar('Blur', 'Camera', 5, 30, update_blur)

# initialize control variables
stored_images = []  # list to store images for stitching
max_images = 3  # number of images to stitch
capture_interval = 5  # time interval between captures in seconds
last_capture_time = datetime.datetime.now()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # display frame
    cv2.imshow('Camera', frame)

    # handle user key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        # capture image if 'c' is pressed
        if len(stored_images) < max_images:
            stored_images.append(frame.copy())
            print("Captured image", len(stored_images))
            if len(stored_images) == max_images:
                # If enough images are captured, stitch them
                panorama = stitch_images(stored_images)
                if panorama is not None:
                    cv2.imshow('Panorama', panorama)
                    cv2.waitKey(0)  # wait for key press to close panorama
                # clear stored images after stitching
                stored_images = []

#cleanup: release camera, close all windows
cap.release()
cv2.destroyAllWindows()
