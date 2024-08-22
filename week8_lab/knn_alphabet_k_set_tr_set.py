import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from ucimlrepo import fetch_ucirepo

# Fetch dataset
letter_recognition = fetch_ucirepo(id=59)

# Extract features and targets
X = letter_recognition.data.features.to_numpy().astype(np.float32)
y = letter_recognition.data.targets

# Encode the labels to numeric values
le = LabelEncoder()
y = le.fit_transform(y)

# Define train/test splits and k values
splits = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
k_values = range(1, 10)

# Dictionary to store accuracies for each split and k value
accuracies = {split: [] for split in splits}

# Loop over each train/test split
for split in splits:
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=split, stratify=y)
    
    # Debugging statements
    print(f'\nTrain/Test Split: {int(split*100)}/{int((1-split)*100)}')
    print(f'Train data shape: {X_train.shape}, Train labels shape: {y_train.shape}')
    print(f'Test data shape: {X_test.shape}, Test labels shape: {y_test.shape}')
    
    # Loop over each k value
    for k in k_values:
        # Initiate kNN, train it on the training data, then test it with the test data
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        accuracy = knn.score(X_test, y_test) * 100.0
        accuracies[split].append(accuracy)
        print(f'k={k}, Accuracy: {accuracy:.2f}%')

# Plot the accuracies
plt.figure(figsize=(12, 8))
for split in splits:
    plt.plot(k_values, accuracies[split], marker='o', linestyle='-', label=f'Train/Test Split {int(split*100)}/{int((1-split)*100)}')

plt.title('k-NN Accuracy for Different k Values and Train/Test Splits (Alphabet Dataset)')
plt.xlabel('k Value')
plt.ylabel('Accuracy (%)')
plt.xticks(k_values)
plt.legend()
plt.grid(True)
plt.show()
