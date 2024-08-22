import numpy as np
import cv2 as cv
import glob

# Termination criteria for corner sub-pixel accuracy
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Update the image folder path and the glob pattern for new image names
image_folder = r'C:\Users\Gio\Desktop\tinyml'
image_files = glob.glob(image_folder + '\\right*.jpg')

if not image_files:
    print("No images found. Check the path and filename pattern.")
else:
    print(f"Found {len(image_files)} images for calibration.")

# Loop over a range of possible chessboard sizes
for width in range(7, 12):
    for height in range(11, 12):
        print(f"Trying chessboard size: {width}x{height}")
        objp = np.zeros((width * height, 3), np.float32)
        objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)
        
        objpoints = []
        imgpoints = []
        
        for fname in image_files:
            img = cv.imread(fname)
            if img is None:
                print(f"Failed to load {fname}")
                continue
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            ret, corners = cv.findChessboardCorners(gray, (width, height), None)
            
            if ret:
                objpoints.append(objp)
                corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)
                cv.drawChessboardCorners(img, (width, height), corners2, ret)
                cv.imshow('Chessboard', img)
                cv.waitKey(500)
            else:
                print(f"Chessboard not found in {fname} for size {width}x{height}")
        
        cv.destroyAllWindows()
        
        # Proceed with calibration if corners were found
        if objpoints and imgpoints:
            print(f"Successful detection with chessboard size: {width}x{height}")
            ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
            print("Camera Matrix:\n", camera_matrix)
            print("Distortion Coefficients:\n", dist_coeffs)

            # Calculate re-projection error
            total_error = 0
            for i in range(len(objpoints)):
                imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
                error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
                total_error += error

            mean_error = total_error / len(objpoints)
            print("Re-projection error:", mean_error)

            # Save the calibration matrix, distortion coefficients, and re-projection error
            np.save('camera_matrix_part2.npy', camera_matrix)
            np.save('dist_coeffs_part2.npy', dist_coeffs)
            np.save('reprojection_error_part2.npy', mean_error)
            
            break  # Optional: break if the first successful calibration is desired
    if objpoints and imgpoints:
        break  # Stop further sizes if successful with current size
