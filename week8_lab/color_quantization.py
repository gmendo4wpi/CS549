import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# Read the image
img = cv.imread(r'C:\Users\Gio\Desktop\tinyml\nature.png')

# Convert the image to a 2D array of pixels
Z = img.reshape((-1, 3))

# Convert to np.float32
Z = np.float32(Z)

# Define criteria, number of clusters(K), and apply kmeans()
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
k_values = [2, 3, 5, 10, 20, 40]

# Create a figure to plot the results
plt.figure(figsize=(15, 10))

for i, K in enumerate(k_values):
    ret, label, center = cv.kmeans(Z, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

    # Convert back to uint8 and reshape to original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    # Plot the quantized image
    plt.subplot(2, 3, i + 1)
    plt.imshow(cv.cvtColor(res2, cv.COLOR_BGR2RGB))
    plt.title(f'K = {K}')
    plt.axis('off')

plt.suptitle('Color Quantization with Different K values')
plt.show()
