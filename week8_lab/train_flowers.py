import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Dense, Flatten
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set image_size and batch_size
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# Target Directory
directory = "C:/Users/Gio/Desktop/tinyml/archive/flowers"

# Train Data
train_data = tf.keras.preprocessing.image_dataset_from_directory(
             directory,
             subset='training',
             validation_split=0.2,
             image_size=IMAGE_SIZE,
             batch_size=BATCH_SIZE,
             seed=42)

# Valid data
valid_data = tf.keras.preprocessing.image_dataset_from_directory(
            directory,
            subset='validation',
            validation_split=0.2,
            image_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            seed=42)

# Class names
class_names = train_data.class_names
print("Class names:", class_names)

# Map preprocess_image function to datasets
def preprocess_image(image, label, image_shape=224):
    img = tf.image.resize(image, [image_shape, image_shape])
    img = img / 255.0
    return tf.cast(img, tf.float32), label

train_data = train_data.map(map_func=preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
valid_data = valid_data.map(map_func=preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

# Shuffle and prefetch datasets
train_data = train_data.shuffle(buffer_size=1000).prefetch(buffer_size=tf.data.AUTOTUNE)
valid_data = valid_data.shuffle(buffer_size=1000).prefetch(buffer_size=tf.data.AUTOTUNE)

# Set random seed
tf.random.set_seed(42)

# Model creation
model_1 = Sequential([
    Conv2D(filters=32, kernel_size=4, padding='same', activation='relu', input_shape=(224, 224, 3)),
    MaxPool2D(2, 2),
    Conv2D(filters=64, kernel_size=4, padding='same', activation='relu'),
    MaxPool2D(2, 2),
    Conv2D(filters=64, kernel_size=4, padding='same', activation='relu'),
    MaxPool2D(2, 2),
    Dropout(0.5),
    Flatten(),
    Dense(len(class_names), activation='softmax')
])

# Compile model
model_1.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
               optimizer='adam',
               metrics=['accuracy'])

# Fit model
history_1 = model_1.fit(train_data,
                       epochs=10,
                       validation_data=valid_data)

# Save model in the new Keras format
model_1.save('C:/Users/Gio/Desktop/tinyml/archive/flower_classifier_model_v2.keras')

# Model summary
model_1.summary()

# Plot loss and accuracy
def plot_loss_curves(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    epochs = range(len(history.history['loss']))

    plt.plot(epochs, loss, label='training_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    
    plt.legend()

    plt.figure()
    plt.plot(epochs, accuracy, label='training_accuracy')
    plt.plot(epochs, val_accuracy, label='val_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()

plot_loss_curves(history=history_1)
