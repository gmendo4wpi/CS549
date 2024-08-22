import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def apply_harris(image):
    # Convert image to grayscale because cornerHarris
    # requires a single channel image format
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    # Apply Harris Corner Detection algorithm
    # parameters include:
    # - block size: size of neighborhood considered for corner detection
    # - ksize: aperture parameter for Sobel operator
    # - k: Harris detector free parameter
    dst = cv.cornerHarris(gray, 2, 3, 0.04)
    # Dilate corner points to enhance corners
    dst = cv.dilate(dst, None)
    result = image.copy()
    result[dst > 0.01 * dst.max()] = [0, 0, 255]
    return result

def apply_sift(image):
    # Convert image to grayscale as SIFT works on single channel images
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # Initialize SIFT detector
    sift = cv.SIFT_create()
    # Detect SIFT keypoints and compute descriptors for grayscale image
    # function returns keypoints and descriptors, here descriptors are unused
    keypoints, _ = sift.detectAndCompute(gray, None)
    # Draw keypoints on original image
    # keypoints: keypoints detected by SIFT
    # color: color of the keypoints (green in this case)
    result = cv.drawKeypoints(image, keypoints, None, color=(0, 255, 0))
    return result

# Load original image
image_path = r'C:\Users\Gio\Downloads\UnityHall.png'
image = cv.imread(image_path)
if image is None:
    print("Error: Image could not be read.")
    exit()

# Define transformations
transformations = {
    'Original': image,
    'Rotated': cv.warpAffine(image, cv.getRotationMatrix2D((image.shape[1] // 2, image.shape[0] // 2), -10, 1), (image.shape[1], image.shape[0])),
    'Scaled Up': cv.resize(image, None, fx=1.2, fy=1.2, interpolation=cv.INTER_CUBIC),
    'Scaled Down': cv.resize(image, None, fx=0.8, fy=0.8, interpolation=cv.INTER_AREA),
    'Affine': cv.warpAffine(image, cv.getAffineTransform(np.float32([[50, 50], [200, 50], [50, 200]]), np.float32([[10, 100], [200, 50], [100, 250]])), (image.shape[1], image.shape[0])),
    'Perspective': cv.warpPerspective(image, cv.getPerspectiveTransform(np.float32([[56, 65], [368, 52], [28, 387], [389, 390]]), np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])), (300, 300))
}

# Apply Harris and SIFT to each transformed image and display results
for title, img in transformations.items():
    harris_img = apply_harris(img)
    sift_img = apply_sift(img)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(cv.cvtColor(harris_img, cv.COLOR_BGR2RGB))
    plt.title(f'Harris Corners - {title}')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv.cvtColor(sift_img, cv.COLOR_BGR2RGB))
    plt.title(f'SIFT Features - {title}')
    plt.axis('off')

    plt.show()
