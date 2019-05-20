from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

"""
Steps:
1. Convolution: we create many feature maps to obtain our first convolution layer
2. Max Pooling
3. Flattening
4. Full connection

"""

# Initialize the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))  # 32 feature detectors with 3*3 kernel size

# Step 2 - Max Pooling
#   Slide with a size of 2
classifier.add(MaxPooling2D)