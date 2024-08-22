import cv2 as cv
import matplotlib.pyplot as plt

# load the images
img1 = cv.imread('oreo.jpg', cv.IMREAD_GRAYSCALE) 
img2 = cv.imread('oreos.jpg', cv.IMREAD_GRAYSCALE)

#check if images are loaded properly
if img1 is None or img2 is None:
    print("Error loading images. Check the file paths.")
    exit()

# initialize SURF detector
minHessian = 400
surf = cv.xfeatures2d.SURF_create(hessianThreshold=minHessian)

# detect keypoints and compute descriptors
kp1, des1 = surf.detectAndCompute(img1, None)
kp2, des2 = surf.detectAndCompute(img2, None)

# create a Brute-Force matcher with default params
bf = cv.BFMatcher()

# match descriptors
matches = bf.match(des1, des2)

# sort matches in the order of their distance
matches = sorted(matches, key=lambda x: x.distance)

# draw the first 10 matches
img_matches = cv.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# display matches
plt.figure(figsize=(12, 6))
plt.imshow(img_matches)
plt.title('Top 10 Matches with SURF and Brute-Force')
plt.show()
