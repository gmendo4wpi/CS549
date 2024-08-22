import cv2 as cv
import matplotlib.pyplot as plt

#load the images
img1 = cv.imread('oreo.jpg', cv.IMREAD_GRAYSCALE) 
img2 = cv.imread('oreos.jpg', cv.IMREAD_GRAYSCALE)

# check if images are loaded properly
if img1 is None or img2 is None:
    print("Error loading images. Check the file paths.")
    exit()

# initialize the SURF detector
minHessian = 400
surf = cv.xfeatures2d.SURF_create(hessianThreshold=minHessian)

# detect keypoints and compute descriptors
kp1, des1 = surf.detectAndCompute(img1, None)
kp2, des2 = surf.detectAndCompute(img2, None)

# set FLANN parameters
index_params = dict(algorithm = 1, trees = 5)  # FLANN_INDEX_KDTREE
search_params = dict(checks=50)  # or pass empty dictionary

# initialize and apply FLANN matcher
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# apply Lowe's ratio test to filter matches
good_matches = []
for m, n in matches:
    if m.distance < 0.55 * n.distance:
        good_matches.append(m)

# draw good matches
img_matches = cv.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# display the matches
plt.figure(figsize=(12, 6))
plt.imshow(img_matches)
plt.title('Good Matches with Ratio Test (SURF and FLANN)')
plt.show()
