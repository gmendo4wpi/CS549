import cv2
import numpy as np
import matplotlib.pyplot as plt

# load image
image_path = r'C:\Users\G\texas.png'
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# apply Gaussian blur
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# apply Canny edge detection
edges = cv2.Canny(blurred, 50, 150)

# detect lines using Hough Transform
lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

# Formulate problem
A = []
b = []

for line in lines:
    rho, theta = line[0]
    A.append([np.cos(theta), np.sin(theta)])
    b.append(rho)

A = np.array(A)
b = np.array(b)

# solve for vanishing point using least squares
AtA = np.dot(A.T, A)
Atb = np.dot(A.T, b)
vanishing_point = np.linalg.solve(AtA, Atb)

# visualize result
u, v = vanishing_point
u, v = int(round(u)), int(round(v))

plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# plot detected lines within image bounds
height, width = image.shape[:2]
for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    plt.plot([x1, x2], [y1, y2], 'r')

# mark vanishing point
plt.scatter([u], [v], c='red', s=100)

# adjust plot limits and aspect ratio
plt.xlim([0, width])
plt.ylim([height, 0])  # invert y-axis to match image coordinates
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
