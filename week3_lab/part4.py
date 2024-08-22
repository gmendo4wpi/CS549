import cv2 as cv
import matplotlib.pyplot as plt

# load images
img1 = cv.imread('oreo.jpg', cv.IMREAD_GRAYSCALE) 
img2 = cv.imread('oreos.jpg', cv.IMREAD_GRAYSCALE) 

# check if images are loaded properly
if img1 is None or img2 is None:
    print("Error loading images. Check the file paths.")
    exit()

# initialize SIFT detector
sift = cv.SIFT_create()

# detect keypoints and compute descriptors
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# create a Brute-Force matcher without crossCheck
bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False)

# perform KNN matching
matches = bf.knnMatch(des1, des2, k=2)

# apply Lowe's ratio test to filter matches
good_matches = []
for m, n in matches:
    if m.distance < 0.55 * n.distance:
        good_matches.append(m)

#draw good matches
img_matches = cv.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# display matches
plt.figure(figsize=(12, 6))
plt.imshow(img_matches)
plt.title('SIFT and Brute-Force')
plt.show()
