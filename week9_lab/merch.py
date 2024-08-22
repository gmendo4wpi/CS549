import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from google.colab import drive
drive.mount('/content/drive')

# set path to dataset
base_dir = '/content/drive/MyDrive/Colab Notebooks/MerchData'

# set up data augmentation and preprocessing using ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1./255,  # normalize pixel values
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  #split data into 80% train and 20% validation
)

# set batch size
batch_size = 32

# set image size to match input size of models
image_size = (224, 224)  # VGG-19 and Inception-V3 both use 224x224

# load images from directories and apply preprocessing and augmentation setup
train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'  # set as training data
)

validation_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'  # set as validation data
)

from tensorflow.keras.applications import VGG19, InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

# load VGG19 model
vgg19 = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# add a global spatial average pooling layer
x_vgg = GlobalAveragePooling2D()(vgg19.output)

# add a fully connected layer
x_vgg = Dense(1024, activation='relu')(x_vgg)

# add a dropout layer to reduce overfitting for VGG19 only
x_vgg = Dropout(0.5)(x_vgg)

# add a final softmax layer for classification
predictions_vgg = Dense(5, activation='softmax')(x_vgg)

# model we will train
model_vgg19 = Model(inputs=vgg19.input, outputs=predictions_vgg)

from tensorflow.keras.optimizers import Adam

# compile models with updated parameter name for learning rate
model_vgg19.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
model_checkpoint_vgg = ModelCheckpoint('best_model_vgg.keras', save_best_only=True)

# fitting model with proper steps and callbacks for VGG19
history_vgg19 = model_vgg19.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    epochs=10,
    callbacks=[early_stopping, model_checkpoint_vgg]
)

from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

# load InceptionV3 model
inception_v3 = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# Freeze layers
for layer in inception_v3.layers:
    layer.trainable = False

# add a global spatial average pooling layer
x_incep = GlobalAveragePooling2D()(inception_v3.output)

# add a fully connected layer
x_incep = Dense(1024, activation='relu')(x_incep)

# add a final softmax layer for classification
predictions_incep = Dense(5, activation='softmax')(x_incep)

# model we will train
model_inceptionv3 = Model(inputs=inception_v3.input, outputs=predictions_incep)

from tensorflow.keras.optimizers import Adam

# compile model with updated parameter name for learning rate
model_inceptionv3.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# train InceptionV3 model
history_inceptionv3 = model_inceptionv3.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    epochs=10
)
