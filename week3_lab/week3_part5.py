import cv2 as cv
import numpy as np

def detect_and_display(model_image, frame, detector, matcher):
    # detect keypoints and descriptors in both images
    kp1, des1 = detector.detectAndCompute(model_image, None)
    kp2, des2 = detector.detectAndCompute(frame, None)

    # match descriptors
    matches = matcher.knnMatch(des1, des2, k=2)

    # filter matches using Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # draw matches
    if len(good_matches) > 10:  # arbitrary threshold to ensure enough matches
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

        # Compute Homography
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        if M is not None:
            h, w = model_image.shape
            pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1,1,2)
            dst = cv.perspectiveTransform(pts, M)
            frame = cv.polylines(frame, [np.int32(dst)], True, (0,255,0), 3, cv.LINE_AA)

    return frame

# initialize camera
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video device.")
    exit()

# load model image
model_image = cv.imread(r'C:\Users\Gio\Downloads\aleve.png', cv.IMREAD_GRAYSCALE)
if model_image is None:
    print("Error: Could not load model image.")
    exit()

# feature Detector and Matcher
detector = cv.SIFT_create()
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
matcher = cv.FlannBasedMatcher(index_params, search_params)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_detected = detect_and_display(model_image, frame_gray, detector, matcher)
    cv.imshow('Camera', frame_detected)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
