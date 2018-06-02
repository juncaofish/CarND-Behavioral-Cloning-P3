import csv
import cv2
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, Flatten, Dense, Dropout, Lambda, Activation, MaxPooling2D
import matplotlib.pyplot as plt

adjust_angles = [0, 0.25, -0.25]


def load_data():
    """
    Load images and angles data collected in training mode.
    :return:
    """
    lines = []
    with open("data/driving_log.csv", "r") as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    images = []
    measurements = []
    for line in lines:
        for i in range(3):
            source_path = line[i]
            filename = source_path.split("/")[-1]
            current_path = "data/IMG/" + filename
            image = cv2.imread(current_path)
            images.append(preprocess(image))
            measurement = float(line[3]) + adjust_angles[i]
            measurements.append(measurement)
    X_train = np.array(images)
    Y_train = np.array(measurements)
    return X_train, Y_train


def preprocess(image, verbose=False):
    """
    Perform preprocessing steps on a single bgr frame.
    These inlcude: cropping, resizing, eventually converting to grayscale
    :param image: input color frame in BGR format
    :param verbose: if true, open debugging visualization
    :return:
    """

    # crop image (remove useless information)
    frame_cropped = image[50:140, :, :]

    # resize image
    frame_resized = cv2.resize(frame_cropped, dsize=(200, 66))
    if verbose:
        plt.figure(1), plt.imshow(cv2.cvtColor(image, code=cv2.COLOR_BGR2RGB))
        plt.figure(2), plt.imshow(cv2.cvtColor(frame_cropped, code=cv2.COLOR_BGR2RGB))
        plt.figure(3), plt.imshow(cv2.cvtColor(frame_resized, code=cv2.COLOR_BGR2RGB))
        plt.show()

    return frame_resized.astype('float32')


def get_nvidia_model():
    """
    :param summary: show model summary
    :return: keras Model of NVIDIA architecture
    """
    init = 'glorot_uniform'
    input_frame = Input(shape=(66, 200, 3))

    # standardize input
    x = Lambda(lambda z: z / 127.5 - 1.)(input_frame)

    x = Convolution2D(24, 5, 5, border_mode='valid', subsample=(2, 2), init=init, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Convolution2D(36, 5, 5, border_mode='valid', subsample=(2, 2), init=init, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Convolution2D(48, 5, 5, border_mode='valid', subsample=(2, 2), init=init, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Convolution2D(64, 3, 3, border_mode='valid', init=init, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Convolution2D(64, 3, 3, border_mode='valid', init=init, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Flatten()(x)

    x = Dense(100, init=init)(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(50, init=init)(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(10, init=init)(x)
    x = Activation('relu')(x)
    out = Dense(1, init=init)(x)

    model = Model(input=input_frame, output=out)
    return model

def get_lenet_model():
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5-1, input_shape=(66,200,3)))
    model.add(Convolution2D(6,5,5,activation="relu"))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6,5,5,activation="relu"))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))
    return model

if __name__ == '__main__':

    # split udacity csv data into training and validation
    X_train, Y_train = load_data()

    # get network model and compile it (default Adam opt)
    # nvidia_net = get_nvidia_model()
    # nvidia_net.compile(optimizer='adam', loss='mse')
    # nvidia_net.summary()
    # nvidia_net.fit(X_train, Y_train, shuffle=True, validation_split=0.2, nb_epoch=3)
    # nvidia_net.save("model.h5")

    lenet = get_lenet_model()
    lenet.compile(optimizer='adam', loss='mse')
    lenet.summary()
    lenet.fit(X_train, Y_train, shuffle=True, validation_split=0.2, nb_epoch=3)
    lenet.save("model1.h5")