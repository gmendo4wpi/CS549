import cv2 as cv
import numpy as np

#input images
def create_panorama(img1_path, img2_path):
    img1 = cv.imread(img1_path)  
    img2 = cv.imread(img2_path)  

    # convert images to grayscale for feature detection
    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    # initialize SIFT detector
    sift = cv.SIFT_create()

    # find keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # store all good matches as per Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    img_matches = cv.drawMatches(gray1, kp1, gray2, kp2, good_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # display matches
    cv.imshow("Matches", img_matches)
    cv.waitKey(0)
    cv.destroyAllWindows()

    if len(good_matches) > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

        # compute Homography
        M, mask = cv.findHomography(dst_pts, src_pts, cv.RANSAC, 5.0)

        # warp second image using computed homography
        h1, w1, _ = img1.shape
        h2, w2, _ = img2.shape

        # calculate size of output panorama
        panorama_width = w1 + w2
        panorama_height = max(h1, h2)

        # warp second image
        img2_warped = cv.warpPerspective(img2, M, (panorama_width, panorama_height))

        # copy img1 on top of warped image
        panorama = np.zeros((panorama_height, panorama_width, 3), dtype=np.uint8)
        panorama[:h1, :w1, :] = img1
        mask = (img2_warped > 0).astype(np.uint8)
        panorama[mask == 1] = img2_warped[mask == 1]

        return panorama
    else:
        print("Not enough matches are found - {}/{}".format(len(good_matches), 10))
        return None

img1_path = r'C:\Users\Gio\Downloads\boston1.jpeg'  
img2_path = r'C:\Users\Gio\Downloads\boston2.jpeg' 

# create panorama
result = create_panorama(img1_path, img2_path)

# display result
if result is not None:
    cv.imshow('Panorama', result)
    cv.waitKey(0)
    cv.destroyAllWindows()
else:
    print("Failed to create a panorama.")
