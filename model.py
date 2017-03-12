import csv
import random
import cv2
import numpy as np
import keras.callbacks
import keras.backend.tensorflow_backend as KTF
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
import sklearn
from sklearn.model_selection import train_test_split
import tensorflow as tf

def main():
    random.seed(4321)

    samples = read_image_name('./data/driving_log.csv')
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)

    train_generator = make_generator(train_samples, batch_size=32, augment=True)
    validation_generator = make_generator(validation_samples, batch_size=32)

    ch, row, col = 3, 160, 320
    old_session = KTF.get_session()

    with tf.Graph().as_default():
        sess = tf.Session('')
        KTF.set_session(sess)
        KTF.set_learning_phase(1)

        model = Sequential()
        model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=(row, col, ch)))
        model.add(Cropping2D(cropping=((70, 25), (0, 0))))
        model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
        model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
        model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
        model.add(Convolution2D(64, 3, 3, activation='relu'))
        model.add(Convolution2D(64, 3, 3, activation='relu'))
        model.add(Flatten())
        model.add(Dense(100))
        model.add(Dropout(0.25))
        model.add(Dense(50))
        model.add(Dropout(0.25))
        model.add(Dense(10))
        model.add(Dropout(0.25))
        model.add(Dense(1))
        model.summary()

        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

        tb_cb = keras.callbacks.TensorBoard(log_dir='./log', histogram_freq=1)
        cbks = [tb_cb]

        model.fit_generator(train_generator, samples_per_epoch=len(train_samples)*6,
                            callbacks=cbks,
                            validation_data=validation_generator,
                            nb_val_samples=len(validation_samples), nb_epoch=5)

        model.save('model.h5')

    KTF.set_session(old_session)

def read_image_name(log_file):
    samples = []
    with open(log_file) as f:
        reader = csv.reader(f)
        next(reader, None)
        for line in reader:
            samples.append(line)
    return samples

def make_generator(samples, batch_size=32, augment=False):
    augment = augment

    def generator(samples, batch_size=32):
        num_samples = len(samples)
        while 1:
            sklearn.utils.shuffle(samples)
            for offset in range(0, num_samples, batch_size):
                batch_samples = samples[offset:offset+batch_size]

                images = []
                angles = []
                for batch_sample in batch_samples:
                    name = './data/IMG/'+batch_sample[0].split('/')[-1]
                    center_image = cv2.imread(name)
                    images.append(center_image)
                    center_angle = float(batch_sample[3])
                    angles.append(center_angle)

                    if augment:
                        name = './data/IMG/'+batch_sample[1].split('/')[-1]
                        left_image = cv2.imread(name)
                        images.append(left_image)
                        left_angle = float(batch_sample[3]) + 0.05
                        angles.append(left_angle)

                        name = './data/IMG/'+batch_sample[2].split('/')[-1]
                        right_image = cv2.imread(name)
                        images.append(right_image)
                        right_angle = float(batch_sample[3]) - 0.05
                        angles.append(right_angle)

                if augment:
                    images.extend([np.fliplr(image) for image in images])
                    angles.extend([-angle for angle in angles])

                assert(len(images) == len(angles))

                # trim image to only see section with road
                X_train = np.array(images)
                y_train = np.array(angles)
                yield sklearn.utils.shuffle(X_train, y_train)

    return generator(samples, batch_size)

if __name__ == "__main__":
    main()
