import cv2 as cv
import numpy as np

def detect_and_display(model_image, frame, detector, matcher, min_matches=10, ransac_thresh=10.0):
    # detect keypoints and descriptors in both images
    kp1, des1 = detector.detectAndCompute(model_image, None)
    kp2, des2 = detector.detectAndCompute(frame, None)

    if des1 is None or des2 is None:
        return frame

    # match descriptors
    matches = matcher.knnMatch(des1, des2, k=2)

    # filter matches using Lowe's ratio test
    good_matches = []
    for match in matches:
        if len(match) == 2:
            m, n = match
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

    # draw matches if we have enough good matches
    if len(good_matches) >= min_matches:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

        # compute Homography
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, ransac_thresh)
        if M is not None:
            h, w = model_image.shape
            pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1,1,2)
            dst = cv.perspectiveTransform(pts, M)
            frame = cv.polylines(frame, [np.int32(dst)], True, (0,255,0), 3, cv.LINE_AA)
        else:
            print("Homography could not be computed")
    else:
        print(f"Not enough matches found: {len(good_matches)}/{min_matches}")

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
orb = cv.ORB_create(nfeatures=2000)  # Initialize ORB detector with more features
FLANN_INDEX_LSH = 6
index_params = dict(algorithm = FLANN_INDEX_LSH,
                    table_number = 6,  # 12
                    key_size = 12,     # 20
                    multi_probe_level = 1)  # 2
search_params = dict(checks=50)  # or pass empty dictionary
flann = cv.FlannBasedMatcher(index_params, search_params)  # initialize FLANN matcher

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_detected = detect_and_display(model_image, frame_gray, orb, flann, min_matches=15, ransac_thresh=10.0)
    cv.imshow('Camera', frame_detected)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
