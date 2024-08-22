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

# Flatten the cells into 5000 samples, each with 400 features (20x20)
x = x.reshape(-1, 400).astype(np.float32)

# Normalize the data
x /= 255.0

# Create labels for the data
k = np.arange(10)
labels = np.repeat(k, 500)

# Define train/test splits and k values
splits = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
k_values = range(1, 10)

# Dictionary to store accuracies for each split and k value
accuracies = {split: [] for split in splits}

# Loop over each train/test split
for split in splits:
    # Determine the number of training samples per class
    num_train_per_class = int(500 * split)
    num_test_per_class = 500 - num_train_per_class
    
    # Prepare the training and test data
    train_indices = np.hstack([np.arange(i * 500, i * 500 + num_train_per_class) for i in range(10)])
    test_indices = np.hstack([np.arange(i * 500 + num_train_per_class, (i + 1) * 500) for i in range(10)])
    
    train = x[train_indices]
    test = x[test_indices]
    train_labels = labels[train_indices]
    test_labels = labels[test_indices]
    
    # Reshape labels to match OpenCV format
    train_labels = train_labels[:, np.newaxis]
    test_labels = test_labels[:, np.newaxis]
    
    # Debugging statements
    print(f'\nTrain/Test Split: {int(split*100)}/{int((1-split)*100)}')
    print(f'Train data shape: {train.shape}, Train labels shape: {train_labels.shape}')
    print(f'Test data shape: {test.shape}, Test labels shape: {test_labels.shape}')
    
    # Verify first few labels to ensure they are correct
    print('Train labels sample:', train_labels[:10].flatten())
    print('Test labels sample:', test_labels[:10].flatten())
    
    # Loop over each k value
    for i in k_values:
        # Initiate kNN, train it on the training data, then test it with the test data
        knn = cv.ml.KNearest_create()
        knn.train(train, cv.ml.ROW_SAMPLE, train_labels)
        ret, result, neighbours, dist = knn.findNearest(test, k=i)
        
        # Check the accuracy of classification
        matches = result == test_labels
        correct = np.count_nonzero(matches)
        accuracy = correct * 100.0 / result.size
        accuracies[split].append(accuracy)
        print(f'k={i}, Accuracy: {accuracy:.2f}%')

# Plot the accuracies
plt.figure(figsize=(12, 8))
for split in splits:
    plt.plot(k_values, accuracies[split], marker='o', linestyle='-', label=f'Train/Test Split {int(split*100)}/{int((1-split)*100)}')

plt.title('k-NN Accuracy for Different k Values and Train/Test Splits')
plt.xlabel('k Value')
plt.ylabel('Accuracy (%)')
plt.xticks(k_values)
plt.legend()
plt.grid(True)
plt.show()
