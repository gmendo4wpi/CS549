import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize images to a range of 0 to 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# Build model using Sequential API
def build_sequential_model():
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(128, activation='relu'),
        layers.Dense(10)
    ])
    return model

# Compile and train model
def compile_and_train_model(model, optimizer, epochs=5):
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    history = model.fit(train_images, train_labels, epochs=epochs, validation_split=0.1)
    return history

# Evaluate model
def evaluate_model(model):
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    return test_loss, test_acc

# Plot training history
def plot_history(history, optimizer_name):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title(f'Training and validation accuracy with {optimizer_name}')
    plt.legend()
    plt.show()

# Main function to train and evaluate model with different optimizers
def main():
    optimizers = ['adam', 'sgd', 'rmsprop']
    results = {}

    for optimizer in optimizers:
        print(f"\nTraining with optimizer: {optimizer}")
        model = build_sequential_model()
        history = compile_and_train_model(model, optimizer)
        test_loss, test_acc = evaluate_model(model)
        results[optimizer] = (test_loss, test_acc)
        plot_history(history, optimizer)

    for optimizer, (loss, acc) in results.items():
        print(f"Optimizer: {optimizer} - Test Loss: {loss:.4f}, Test Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
