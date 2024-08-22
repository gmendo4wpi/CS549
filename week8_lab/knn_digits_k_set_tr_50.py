import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# Load the digits image
img = cv.imread(r'C:\Users\Gio\Desktop\tinyml\digits.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Split the image to 5000 cells, each 20x20 size
cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]

# Make it into a Numpy array: its size will be (50,100,20,20)
x = np.array(cells)

# Prepare the training data and test data
train = x[:, :50].reshape(-1, 400).astype(np.float32)  # Size = (2500,400)
test = x[:, 50:100].reshape(-1, 400).astype(np.float32)  # Size = (2500,400)

# Create labels for train and test data
k = np.arange(10)
train_labels = np.repeat(k, 250)[:, np.newaxis]
test_labels = train_labels.copy()

# List to store accuracies for each k
accuracies = []

# Loop over different k values
for i in range(1, 10):
    # Initiate kNN, train it on the training data, then test it with the test data
    knn = cv.ml.KNearest_create()
    knn.train(train, cv.ml.ROW_SAMPLE, train_labels)
    ret, result, neighbours, dist = knn.findNearest(test, k=i)
    
    # Check the accuracy of classification
    matches = result == test_labels
    correct = np.count_nonzero(matches)
    accuracy = correct * 100.0 / result.size
    accuracies.append(accuracy)
    print(f'k={i}, Accuracy: {accuracy:.2f}%')

# Plot the accuracies
plt.figure(figsize=(10, 6))
plt.plot(range(1, 10), accuracies, marker='o', linestyle='-', color='b')
plt.title('k-NN Accuracy for Different k Values')
plt.xlabel('k Value')
plt.ylabel('Accuracy (%)')
plt.xticks(range(1, 10))
plt.grid(True)
plt.show()
