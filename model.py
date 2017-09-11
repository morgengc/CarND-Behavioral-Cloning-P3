import csv
import cv2
import numpy as np
import sklearn
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

DRIVING_LOG_CSV = "./data/driving_log.csv"
IMAGE_PATH = "./data/IMG/"
STEERING = 0.22
FLIP_PROBILITY = 0.5
NUMBER_EPOCH = 3


def load_data(csv_path):
    samples = []
    with open(csv_path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
    del samples[0]
    return samples


def generator(samples, batch_size=32, training=True):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                # center, left, right => 0, 1, 2
                if training:
                    camera_source = np.random.randint(0, 3)
                else:
                    camera_source = 0
                name = IMAGE_PATH + batch_sample[camera_source].split('/')[-1]
                image = cv2.imread(name)
                angle = float(batch_sample[3])

                # Make angle correction if not center camera
                if camera_source == 1:
                    angle += STEERING
                if camera_source == 2:
                    angle -= STEERING

                # Flip image randomly
                if training and np.random.rand() > FLIP_PROBILITY:
                    angle *= -1.0
                    image = cv2.flip(image, 1)

                images.append(image)
                angles.append(angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


samples = load_data(DRIVING_LOG_CSV)
samples = shuffle(samples)
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

train_generator = generator(train_samples)
validation_generator = generator(validation_samples, training=False)

model = Sequential()
model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator,
                    samples_per_epoch=32000,
                    validation_data=validation_generator,
                    nb_val_samples=len(validation_samples),
                    nb_epoch=NUMBER_EPOCH)
model.save('model.h5')

