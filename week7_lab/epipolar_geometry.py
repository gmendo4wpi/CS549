import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def draw_epipolar_lines(file1, file2):
    # Load images in grayscale
    img1 = cv.imread(file1, cv.IMREAD_GRAYSCALE)
    img2 = cv.imread(file2, cv.IMREAD_GRAYSCALE)

    # Initialize SIFT detector
    sift = cv.SIFT_create()

    # Detect keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # FLANN parameters and matcher setup
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=55)  # or use empty dictionary
    #search_params = dict(checks=70)  # or use empty dictionary
    flann = cv.FlannBasedMatcher(index_params, search_params)
    
    matches = flann.knnMatch(des1, des2, k=2)

    # Lowe's ratio test to find good matches
    good = []
    for m, n in matches:
        if m.distance < 0.97 * n.distance:
            good.append(m)

    # Extract location of good matches
    points1 = np.float32([kp1[m.queryIdx].pt for m in good])
    points2 = np.float32([kp2[m.trainIdx].pt for m in good])

    # Compute Fundamental Matrix
    F, mask = cv.findFundamentalMat(points1, points2, cv.FM_LMEDS)

    # We select only inlier points
    points1 = points1[mask.ravel() == 1]
    points2 = points2[mask.ravel() == 1]

    def drawlines(img1, img2, lines, pts1, pts2):
        ''' img1 - image on which we draw the epilines for the points in img2
            lines - corresponding epilines '''
        r, c = img1.shape
        img1_color = cv.cvtColor(img1, cv.COLOR_GRAY2BGR)
        img2_color = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)
        for r, pt1, pt2 in zip(lines, pts1, pts2):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            x0, y0 = map(int, [0, -r[2] / r[1]])
            x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
            # Convert points to integers
            pt1 = (int(pt1[0]), int(pt1[1]))
            pt2 = (int(pt2[0]), int(pt2[1]))
            img1_color = cv.line(img1_color, (x0, y0), (x1, y1), color, 1)
            img1_color = cv.circle(img1_color, pt1, 5, color, -1)
            img2_color = cv.circle(img2_color, pt2, 5, color, -1)
        return img1_color, img2_color


    # Find epilines
    lines1 = cv.computeCorrespondEpilines(points2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    img5, img6 = drawlines(img1, img2, lines1, points1, points2)

    lines2 = cv.computeCorrespondEpilines(points1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    img3, img4 = drawlines(img2, img1, lines2, points2, points1)

    plt.figure(figsize=(10, 8))
    plt.subplot(121), plt.imshow(img5)
    plt.subplot(122), plt.imshow(img3)
    plt.show()

# File paths adjusted to your specified directory
file_left = r'C:\Users\G\globe_left.jpg'
file_center = r'C:\Users\G\globe_center.jpg'
file_right = r'C:\Users\G\globe_right.jpg'

draw_epipolar_lines(file_left, file_center)
draw_epipolar_lines(file_center, file_right)


def load_images():
    img1 = cv.imread('globe_left.jpg', cv.IMREAD_GRAYSCALE)  # query image
    img2 = cv.imread('globe_center.jpg', cv.IMREAD_GRAYSCALE)  # train image
    img3 = cv.imread('globe_right.jpg', cv.IMREAD_GRAYSCALE)  # additional image for the second pair
    return img1, img2, img3

def drawlines(img1, img2, lines, pts1, pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r, c = img1.shape
    img1 = cv.cvtColor(img1, cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1 = cv.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2