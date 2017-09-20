import csv
import cv2
import numpy as np
import sys
from random import shuffle
import sklearn


data_size_limit = 9999999
use_generator = False
use_drop_out = True
nb_epoch = 5

def generator(samples, batch_size=32):
  num_samples = len(samples)
  while 1: # Loop forever so the generator never terminates
    shuffle(samples)
    for offset in range(0, num_samples, batch_size):
      batch_samples = samples[offset:offset+batch_size]
      images = []
      angles = []
      for batch_sample in batch_samples:
        name = batch_sample[0]
        center_image = cv2.imread(name)
        center_angle = float(batch_sample[3])
        images.append(center_image)
        angles.append(center_angle)
        images.append(np.fliplr(center_image))
        measurements.append(-center_angle)

      X_train = np.array(images)
      y_train = np.array(angles)
      yield sklearn.utils.shuffle(X_train, y_train)

def GetLines(path, relative_path):
  lines = []
  with open(path) as csvfile:
    reader = csv.reader(csvfile)
    size = 0
    for line in reader:
      size+=1
      if size <= 1:
        continue
      filename = line[0].split('/')[-1]
      line[0] = relative_path + filename
      lines.append(line)
  return lines

lines = []
lines = GetLines('../sim-data/driving_log.csv', '../sim-data/IMG/')
lines.extend(GetLines('../sim-more-data-reverse/driving_log.csv', '../sim-more-data-reverse/IMG/'))
lines.extend(GetLines('../sim-off-road-data-1/driving_log.csv', '../sim-off-road-data-1/IMG/'))
lines.extend(GetLines('../sim-off-road-data-2/driving_log.csv', '../sim-off-road-data-2/IMG/'))
#lines.extend(GetLines('../sim-more-data/driving_log.csv', '../sim-more-data/IMG/'))
lines.extend(GetLines('../sim-off-road-data-3/driving_log.csv', '../sim-off-road-data-3/IMG/'))
lines.extend(GetLines('../sim-2-laps/driving_log.csv', '../sim-2-laps/IMG/'))
print ("total lines %d " % len(lines))

train_samples = []
validation_samples = []
images = []
measurements = []
def ReadAll(images, measurements):
  size = 0;
  for line in lines:
    size += 1
    sc = line[0]
    path = sc
    image = cv2.imread(path)
    if size > data_size_limit:
      break
    images.append(image)
    measurements.append(float(line[3]))
    images.append(np.fliplr(image))
    measurements.append(-float(line[3]))

if use_generator:
  from sklearn.model_selection import train_test_split
  train_samples, validation_samples = train_test_split(lines, test_size=0.2)
  train_generator = generator(train_samples, batch_size=32)
  validation_generator = generator(validation_samples, batch_size=32)
else:
  ReadAll(images, measurements)
  print (len(images))
  x_train = np.array(images)
  y_train = np.array(measurements)
  print(x_train.shape)


from keras.models import Sequential
from keras.layers import Dense, Lambda, Activation, Flatten, Dropout, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x : (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((75,25), (0,0))))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(48, 5, 5, activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
drop_out_path = "_no_dropout"
if use_drop_out:
  model.add(Dropout(0.5))
  model.add(Activation('relu'))
  drop_out_path = "_dropout"
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
if use_generator:
  model.fit_generator(train_generator, samples_per_epoch=
                      len(train_samples), validation_data=validation_generator,
                      nb_val_samples=len(validation_samples), nb_epoch= nb_epoch)
else:
  model.fit(x_train, y_train, validation_split=0.2, nb_epoch=nb_epoch, shuffle=True)
model.save('model_v2.2_less'+ drop_out_path + '.h5')
