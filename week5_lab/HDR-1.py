import cv2 as cv
import numpy as np

# define paths to images
image_paths = (
    r'C:\Users\Gio\Downloads\IMAGE_1.jpg',
    r'C:\Users\Gio\Downloads\IMAGE_2.jpg',
    r'C:\Users\Gio\Downloads\IMAGE_3.jpg'
)

# load images from specified paths
images = [cv.imread(path) for path in image_paths]

# check if all images are loaded properly
if any(img is None for img in images):
    raise ValueError("One or more images could not be loaded. Please check the file paths.")

# initialize Mertens Fusion algorithm
merge_mertens = cv.createMergeMertens()

# process images using Mertens Fusion algorithm
hdr_image = merge_mertens.process(images)

# convert floating-point HDR image to an 8-bit format for visualization
hdr_image_8bit = np.clip(hdr_image * 255, 0, 255).astype('uint8')

# specify output path for HDR image
output_path = r'C:\Users\Gio\Downloads\hdr_result.png'

# save HDR image to output path
cv.imwrite(output_path, hdr_image_8bit)

# print output path (optional)
print("HDR image saved to:", output_path)
