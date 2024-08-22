import cv2
import numpy as np

# load an image from specified path and convert to grayscale
def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Unable to load image at {path}")
    return img

# detect circles for quarters using hough transform
def detect_quarter_circles(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    return cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=20, minRadius=49, maxRadius=50)

# detect circles for nickels using hough transform
def detect_nickel_circles(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    return cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=20, minRadius=40, maxRadius=45)

# detect circles for pennies using hough transform
def detect_penny_circles(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    return cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=20, minRadius=35, maxRadius=38)

# detect circles for dimes using hough transform
def detect_dime_circles(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    return cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=20, minRadius=29, maxRadius=30)

# initialize sift feature detector
def initialize_detector():
    return cv2.SIFT_create()

# find keypoints and descriptors in an image using a detector
def find_keypoints_and_descriptors(image, detector):
    keypoints, descriptors = detector.detectAndCompute(image, None)
    return keypoints, descriptors

# match descriptors between two images using flann based matcher
def match_descriptors(detector, desc1, desc2):
    flann = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 5}, {'checks': 50})
    matches = flann.knnMatch(desc1, desc2, k=2)
    return len([m for m, n in matches if m.distance < 0.75 * n.distance])

# process each frame to detect coins, identify their type, and draw circles and labels
def process_frame(frame, coin_descriptors, detector, total_value, coin_flags, coin_values):
    colors = {
        "Quarter": (0, 255, 0),
        "Nickel": (255, 0, 0),
        "Penny": (255, 255, 0),
        "Dime": (0, 0, 255)
    }
    for coin_type, desc in coin_descriptors.items():
        circles = globals()[f"detect_{coin_type.lower()}_circles"](frame)
        if circles is not None:
            circles = np.uint16(np.around(circles[0]))
            for circle in circles:
                x, y, r = circle
                roi = frame[y-r:y+r, x-r:x+r]
                if roi.size == 0 or roi.shape[0] != 2*r or roi.shape[1] != 2*r:
                    continue
                kp, desc = find_keypoints_and_descriptors(roi, detector)
                matches = match_descriptors(detector, desc, coin_descriptors[coin_type])
                if matches > 4:
                    if not coin_flags[coin_type]:  # Only add to total once
                        total_value += coin_values[coin_type]
                        coin_flags[coin_type] = True
                    cv2.circle(frame, (x, y), r, colors[coin_type], 2)
                    cv2.putText(frame, f"{coin_type} Detected", (x - r, y - r - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[coin_type], 2)

    # display total value on frame
    cv2.putText(frame, f"Total Value: ${total_value:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    return frame, total_value

# main function to initialize camera, process input, and display output
def main():
    coin_paths = {
        "Quarter": r'C:\Users\Gio\Downloads\quarter_rr.png',
        "Nickel": r'C:\Users\Gio\Downloads\nickel_rr.png',
        "Penny": r'C:\Users\Gio\Downloads\penny_rr.png',
        "Dime": r'C:\Users\Gio\Downloads\dime_rr.png'
    }
    coin_values = {"Quarter": 0.25, "Nickel": 0.05, "Penny": 0.01, "Dime": 0.10}
    coin_flags = {"Quarter": False, "Nickel": False, "Penny": False, "Dime": False}
    detector = initialize_detector()
    coin_descriptors = {coin: find_keypoints_and_descriptors(load_image(path), detector)[1] for coin, path in coin_paths.items()}

    total_value = 0.0
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame, total_value = process_frame(frame, coin_descriptors, detector, total_value, coin_flags, coin_values)
        cv2.imshow("Real-Time Coin Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
