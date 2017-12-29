import glob
import numpy as np
from PIL import Image
from os import path
log_list = glob.glob("data/**/driving_log.csv")
print(log_list)

driving_log = []
for log in log_list:
    print(log)
    csv = np.genfromtxt(log, dtype = None, delimiter = ",").tolist()
    log_dir = path.dirname(log)
    for i in range(len(csv)):
        csv[i] = list(csv[i])
        for j in range(3):
            image_name = path.basename(csv[i][j].decode())
            image_path = path.join(log_dir, path.join('IMG', image_name))
            csv[i][j] = image_path
    driving_log += csv

np.random.shuffle(driving_log) # shuffle driving log

images = []
measurements = []
width = 320
height = 160
image_size = (height, width, 3)
batch_size = 32
validation_split = 0.2
# implement shuffling (Done)
# implement data augmentation
def data_augmentor(image, steering):
    # flip image and steering with probability 0.5
    if np.random.choice(2) == 0:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        steering = - steering
    return np.array(image), steering

def data_generator(target_measurement = 0, validation_split = validation_split, validation = False): # 0 is steering
    image_batch = []
    measurement_batch = []
    while True:
        train_length = int(len(driving_log)*(1-validation_split))
        if validation:
            data_range = np.arange(train_length, len(driving_log))
        else:
            data_range = np.arange(train_length)
            np.random.shuffle(data_range)

        for i in data_range:
            while i >= len(images):
                n = len(images)
                image_path = driving_log[n][0]
                img = Image.open(image_path)
                img.load()
                # img = img.resize((width, height))
                # images.append(np.array(img).astype(float)/255.0)
                images.append(img)
                measurements.append(driving_log[n][3:])

            image, steering = data_augmentor(images[i], measurements[i][target_measurement])
            image_batch.append(image)
            measurement_batch.append(steering)

            if len(image_batch) == batch_size:
                yield (np.stack(image_batch), np.stack(measurement_batch))
                image_batch = []
                measurement_batch = []

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Lambda, Convolution2D, MaxPooling2D, Cropping2D
from keras import optimizers

model = Sequential()
model.add(Lambda(lambda x: x/255.0-0.5, input_shape = image_size))
model.add(Cropping2D(cropping = ((70,25), (0,0))))
model.add(Convolution2D(12,5,5,activation = 'relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5,activation = 'relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(1))

optimizer = optimizers.Adam(lr=0.001)
model.compile(loss = 'mse', optimizer = optimizer)
model.fit_generator(data_generator(), steps_per_epoch = len(driving_log)*(1-validation_split)/batch_size ,epochs = 7, validation_data = data_generator(validation = True), validation_steps = len(driving_log)*validation_split/batch_size)
model.save("model.h5")
