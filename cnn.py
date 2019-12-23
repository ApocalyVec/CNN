from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
import PIL.Image
import tensorflow

"""
Steps:
1. Convolution: we create many feature maps to obtain our first convolution layer
2. Max Pooling
3. Flattening
4. Full connection

"""

# Part 1 - Building the CNN

# Initialize the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(
    Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))  # 32 feature detectors with 3*3 kernel size

# Step 2 - Max Pooling
#   Sliding window with a size of 2
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# * Step 2.1 - Second convolutional - maxpooling Layer
classifier.add(
    Conv2D(32, (3, 3), activation='relu'))  # we don't need the input_shape because we have layers before this layer
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Step 3 - Flattening
#   this will be the input layer of the classic ANN that immediately follows
classifier.add(Flatten())

# Step 4 - Full connection ()
#   the hidden layer of the classic ANN
classifier.add(Dense(units=128, activation='relu'))
# Output layer
# sigmoid for binary activation, for more than two, we use softmax
classifier.add(Dense(units=1, activation='sigmoid'))

# Compile the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Part 2 - Fitting the CNN to the images
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(training_set,
                         steps_per_epoch=8000,
                         epochs=3,
                         validation_data=test_set,
                         validation_steps=2000)

# save the trained model
from keras.models import load_model

classifier.save('/trained_model/classifier.h5')
'''
use this line if you want to load the model back
classifier = load_model('/trained_model/classifier.h5')
'''
# make a single prediction
import numpy as np
from keras_preprocessing import image

test_image = image.load_img('/dataset/single_prediction/cat_or_dog_1.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = classifier.predict(test_image)
print(str(training_set.class_indices))  # get which prediction result correspond to which classification