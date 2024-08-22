import cv2
import datetime
import numpy as np

# function to update the zoom level based on trackbar movement
def update_zoom(val):
    global zoom_level
    zoom_level = val
    
# function to update the amount of blur based on trackbar movement
def update_blur(val):
    global blur_amount
    blur_amount = val

# function to apply gaussian blur to the frame
def gaussian_blur(frame, sigma):
    if sigma < 1:
        sigma = 1
    return cv2.GaussianBlur(frame, (0, 0), sigmaX=sigma, sigmaY=sigma)

# function to extract blue color from the frame
def extract_color(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([110,50,50])  # blue
    upper_blue = np.array([130,255,255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    return cv2.bitwise_and(frame, frame, mask=mask)

# function to rotate the frame by a given angle
def rotate_image(frame, angle):
    center = (frame.shape[1] // 2, frame.shape[0] // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(frame, rot_mat, (frame.shape[1], frame.shape[0]))

# function to apply a binary threshold to the frame
def threshold_image(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return thresh

# function to sharpen the image by applying a sharpening kernel
def sharpen_image(frame):
    sharpening_kernel = np.array([[-1, -1, -1],
                                  [-1, 9, -1],
                                  [-1, -1, -1]])
    sharpened_frame = cv2.filter2D(frame, -1, sharpening_kernel)
    return sharpened_frame

def update_sobel_kernel_size(val):
    global sobel_kernel_size
    # ensure kernel size is at least 3 and not larger than 31
    if val < 3:
        val = 3
    if val > 31:
        val = 31
    # ensure kernel size is odd
    if val % 2 == 0:
        val += 1
    sobel_kernel_size = val

def update_canny_threshold1(val):
    global canny_threshold1
    canny_threshold1 = val

def update_canny_threshold2(val):
    global canny_threshold2
    canny_threshold2 = val

def custom_sobel_operator(src_gray, kernel_size=3):
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd")

    scale = 1.0 / (2.0 * (kernel_size // 2))
    Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) * scale
    Gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]) * scale
    
    grad_x = cv2.filter2D(src_gray, -1, Gx)
    grad_y = cv2.filter2D(src_gray, -1, Gy)
    
    return cv2.convertScaleAbs(grad_x), cv2.convertScaleAbs(grad_y)

def custom_laplacian_operator(src_gray, kernel_size=3):
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd")

    laplacian_kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]]) * (1.0 / kernel_size)
    dst = cv2.filter2D(src_gray, -1, laplacian_kernel)
    
    return cv2.convertScaleAbs(dst)

# initialize the camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video device.")
    exit()
    
# set camera resolution
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

# create camera window and trackbars for zoom and blur
cv2.namedWindow('Camera')
cv2.createTrackbar('Zoom', 'Camera', 1, 10, update_zoom)
cv2.createTrackbar('Blur', 'Camera', 5, 30, update_blur)

# initialize control variables
zoom_level = 1
blur_amount = 5
angle = 0
apply_color_extraction = False
apply_thresholding = False
apply_blurring = False
apply_sharpening = False
apply_sobel_x = False
apply_sobel_y = False
apply_canny = False
sobel_kernel_size = 10
canny_threshold1 = 100
canny_threshold2 = 200

# Add trackbars in the initialization section
cv2.createTrackbar('Sobel Kernel Size', 'Camera', 10, 50, update_sobel_kernel_size)
cv2.createTrackbar('Canny Threshold 1', 'Camera', 100, 5000, update_canny_threshold1)
cv2.createTrackbar('Canny Threshold 2', 'Camera', 200, 5000, update_canny_threshold2)

# flags for recording video and output video object
recording = False
out = None

# load and configure logo image
logo_path = r'C:\Users\Gio\Downloads\opencv_logo.png'  # correct path to image
logo = cv2.imread(logo_path)
if logo is None:
    print(f"Error: Unable to load image at {logo_path}")
    exit()
logo = cv2.resize(logo, (100, 100))  # resize logo

s_pressed = False  # flag to check if 's' is pressed
apply_custom_filters = False
apply_multi_window = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    # Convert frame to grayscale for gradient operations
    src_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # apply effects based on user input
    if apply_color_extraction:
        frame = extract_color(frame)

    if angle != 0:
        frame = rotate_image(frame, angle)

    if apply_thresholding:
        frame = threshold_image(frame)

    if apply_blurring:
        frame = gaussian_blur(frame, blur_amount)

    if apply_sharpening:
        frame = sharpen_image(frame)

    if apply_sobel_x:
        sobelx = cv2.Sobel(src_gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel_size)
        frame = cv2.convertScaleAbs(sobelx)  # Display Sobel X

    if apply_sobel_y:
        sobely = cv2.Sobel(src_gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel_size)
        frame = cv2.convertScaleAbs(sobely)  # Display Sobel Y

    if apply_canny:
        frame = cv2.Canny(src_gray, canny_threshold1, canny_threshold2)  # Display Canny edges

    # handle user key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('e'):
        apply_color_extraction = not apply_color_extraction
    elif key == ord('r'):
        angle = (angle - 10) % 360
    elif key == ord('t'):
        apply_thresholding = not apply_thresholding
    elif key == ord('b'):
        apply_blurring = not apply_blurring
    elif key == ord('s'):
        apply_sharpening = not apply_sharpening
    elif key == ord('x'):
        apply_sobel_x = not apply_sobel_x  # Toggle Sobel X
    elif key == ord('y'):
        apply_sobel_y = not apply_sobel_y  # Toggle Sobel Y
    elif key == ord('d'):
        apply_canny = not apply_canny  # Toggle Canny edge detection
    elif key == ord('4'):
        apply_multi_window = not apply_multi_window  # Toggle multi-window display

    # display the frame
    cv2.imshow('Camera', frame)
    
# cleanup: release camera and close all windows
cap.release()
cv2.destroyAllWindows()
