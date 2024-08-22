import numpy as np
import cv2 as cv

# Load the image with the correct path
image = cv.imread(r'C:\Users\Gio\Downloads\UnityHall.png')
if image is None:
    print("Error: Image could not be read.")
    exit()

# 1) Rotation by 10 degrees
center = (image.shape[1] // 2, image.shape[0] // 2)
rot_matrix = cv.getRotationMatrix2D(center, -10, 1)  # Negative for clockwise rotation
rotated_image = cv.warpAffine(image, rot_matrix, (image.shape[1], image.shape[0]))

# 2) Scale up by 10%
scaled_up_image = cv.resize(image, None, fx=1.1, fy=1.1, interpolation=cv.INTER_CUBIC)

# 3) Scale down by 50%
scaled_down_image = cv.resize(image, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)

# 4) Affine transformation
pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
affine_matrix = cv.getAffineTransform(pts1, pts2)
affine_image = cv.warpAffine(image, affine_matrix, (image.shape[1], image.shape[0]))

# 5) Perspective transformation
pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
perspective_matrix = cv.getPerspectiveTransform(pts1, pts2)
perspective_image = cv.warpPerspective(image, perspective_matrix, (300, 300))

# Display results
cv.imshow('Original', image)
cv.imshow('Rotated', rotated_image)
cv.imshow('Scaled Up', scaled_up_image)
cv.imshow('Scaled Down', scaled_down_image)
cv.imshow('Affine', affine_image)
cv.imshow('Perspective', perspective_image)

cv.waitKey(0)  # Wait for a key press to close the displayed windows
cv.destroyAllWindows()  # Clean up windows
