import csv
import cv2
import numpy as np
import sys

data_size_limit = 5000

lines = []
with open('../sim-data/driving_log.csv') as csvfile:
  reader = csv.reader(csvfile)
  size = 0
  for line in reader:
    size+=1
    if size <= 1:
      continue
    lines.append(line)

images = []
measurements = []
size = 0;
for line in lines:
  size += 1
  sc = line[0]
  filename = sc.split('/')[-1]
  path = '../sim-data/IMG/' + filename
  image = cv2.imread(path)
  if size > data_size_limit:
    break
  images.append(image)
  measurements.append(float(line[3]))

print (len(images))
x_train = np.array(images)
y_train = np.array(measurements)

print(x_train.shape)

from keras.models import Sequential
from keras.layers import Dense, Lambda, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x : (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
model.add(Convolution2D(6, 5, 5, activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6, 5, 5, activation="relu"))
model.add(MaxPooling2D())
#model.add(Dropout(0.5))
#model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, validation_split=0.2, nb_epoch=5, shuffle=True)

model.save('model.h5')
