import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def create_disparity_map(left_image_path, right_image_path, output_image_path):
    # Load the images in grayscale
    imgL = cv.imread(left_image_path, cv.IMREAD_GRAYSCALE)
    imgR = cv.imread(right_image_path, cv.IMREAD_GRAYSCALE)

    # Check if images loaded properly
    if imgL is None or imgR is None:
        print("Error loading images!")
        return

    # Initialize the stereo block matching object
    stereo = cv.StereoBM_create(numDisparities=32*5, blockSize=27)
    stereo.setTextureThreshold(15)
    stereo.setUniquenessRatio(15)

    # Compute the disparity map
    disparity = stereo.compute(imgL, imgR)
    disparity = cv.medianBlur(disparity, 5)
    cv.filterSpeckles(disparity, 0, 400, 320)

    # Normalize the disparity map for display (optional, enhances visualization)
    norm_disparity = cv.normalize(disparity, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)

    # Display the disparity map
    plt.imshow(norm_disparity, 'gray')
    plt.colorbar()  # Adds a colorbar to illustrate disparity range
    plt.show()

    # Save the disparity map to an image file
    cv.imwrite(output_image_path, norm_disparity)

if __name__ == "__main__":
    # Path to the left and right images
    left_image_path = r'C:\Users\G\aloeL.jpg'
    right_image_path = r'C:\Users\G\aloeR.jpg'
    
    # Path to save the disparity map image
    output_image_path = r'C:\Users\G\disparity.jpg'
    
    # Create and save the disparity map
    create_disparity_map(left_image_path, right_image_path, output_image_path)
