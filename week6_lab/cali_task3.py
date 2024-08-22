import numpy as np
import cv2 as cv
import glob

# Termination criteria for corner sub-pixel accuracy
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points based on the chessboard pattern (7x6 grid)
chessboard_size = (9, 6)
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images
objpoints = []  # 3D points in real-world space
imgpoints = []  # 2D points in image plane

# Load calibration images
image_folder = r'C:\Users\Gio\Desktop\tinyml'
image_files = glob.glob(image_folder + '\custom*.jpg')  # Change the path accordingly

for fname in image_files:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        cv.drawChessboardCorners(img, chessboard_size, corners2, ret)
        cv.imshow('Chessboard', img)
        cv.waitKey(500)

cv.destroyAllWindows()

# Calibrate the camera
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Save the calibration matrix and distortion coefficients
np.save('camera_matrix_part3.npy', camera_matrix)
np.save('dist_coeffs.npy_part3', dist_coeffs)

# Print the calibration matrix and distortion coefficients
print("Camera Matrix:\n", camera_matrix)
print("Distortion Coefficients:\n", dist_coeffs)

# Load a test image
test_image_path = r'C:\Users\Gio\desktop\tinyml\custom01.jpg'  # Change the path to your test image if needed
img = cv.imread(test_image_path)
h, w = img.shape[:2]

# Undistort the image
new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
dst = cv.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)

# Crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('undistorted_image.jpg', dst)

# Calculate re-projection error
total_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
    total_error += error

mean_error = total_error / len(objpoints)
print("Re-projection error:", mean_error)
np.save('reprojection_error_part3.npy', mean_error)
