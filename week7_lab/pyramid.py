import numpy as np
import cv2 as cv

# Hardcoded camera matrix and distortion coefficients
camera_matrix = np.array([[533.82282879, 0, 341.1335799],
                          [0, 533.87400829, 231.98840517],
                          [0, 0, 1]], dtype=float)
dist_coeffs = np.array([[-0.293296556, 0.118893228, 0.00130928547, -0.000193507334, 0.0156555154]], dtype=float)

def draw(img, corners, imgpts, axis_imgpts):
    imgpts = np.int32(imgpts).reshape(-1, 2)
    axis_imgpts = np.int32(axis_imgpts).reshape(-1, 2)

    # Draw the base in green
    img = cv.drawContours(img, [imgpts[:3]], -1, (0, 255, 0), -3)
    # Draw the pillars in red
    img = cv.line(img, tuple(imgpts[0]), tuple(imgpts[3]), (0, 0, 255), 5)
    img = cv.line(img, tuple(imgpts[1]), tuple(imgpts[3]), (0, 0, 255), 5)
    img = cv.line(img, tuple(imgpts[2]), tuple(imgpts[3]), (0, 0, 255), 5)

    # Draw axes
    img = cv.line(img, tuple(axis_imgpts[0]), tuple(axis_imgpts[1]), (0, 0, 255), 5)  # X-axis in red
    img = cv.line(img, tuple(axis_imgpts[0]), tuple(axis_imgpts[2]), (0, 255, 0), 5)  # Y-axis in green
    img = cv.line(img, tuple(axis_imgpts[0]), tuple(axis_imgpts[3]), (255, 0, 0), 5)  # Z-axis in blue

    return img

# Object points corresponding to the selected corners of the chessboard
objp = np.array([[0, 0, 0], [2, 0, 0], [1, np.sqrt(3), 0], [1, np.sqrt(3)/3, 2*np.sqrt(6)/3]], dtype=np.float32)
axis_length = 2  # Length of the axes
axis_points = np.float32([
    [0, 0, 0],             # Origin
    [axis_length, 0, 0],   # X-axis
    [0, axis_length, 0],   # Y-axis
    [0, 0, axis_length]    # Z-axis
])

# Specify the path to your specific image
img_path = r'C:\Users\G\left08.jpg'
img = cv.imread(img_path)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Find the chessboard corners
chessboard_size = (7, 6)
ret, corners = cv.findChessboardCorners(gray, chessboard_size, None)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

if ret:
    corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    imgpoints = corners2[:4].reshape(-1, 2)

    # Find the rotation and translation vectors using solvePnP with P3P algorithm
    ret, rvecs, tvecs = cv.solvePnP(objp, imgpoints, camera_matrix, dist_coeffs, flags=cv.SOLVEPNP_P3P)

    if ret:  # Ensure that solvePnP successfully found the pose
        # Project 3D points to image plane
        imgpts, _ = cv.projectPoints(objp, rvecs, tvecs, camera_matrix, dist_coeffs)
        axis_imgpts, _ = cv.projectPoints(axis_points, rvecs, tvecs, camera_matrix, dist_coeffs)

        img = draw(img, imgpoints, imgpts, axis_imgpts)
        cv.imshow('Projected Tetrahedron with Axes', img)
        k = cv.waitKey(0) & 0xFF
        if k == ord('s'):
            cv.imwrite(r'C:\Users\G\triangular.jpg', img)
    else:
        print("Failed to find pose.")
else:
    print("Chessboard corners not found in the image.")

cv.destroyAllWindows()
