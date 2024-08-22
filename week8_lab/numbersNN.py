import numpy as np
from tensorflow import keras

# Define model
model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

# Compile model
model.compile(optimizer='sgd', loss='mean_squared_error')

# Define training data
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# Train model
model.fit(xs, ys, epochs=500)

# Make a prediction
print(model.predict(np.array([10.0])))
