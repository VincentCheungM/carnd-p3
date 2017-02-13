import tensorflow as tf
from keras.layers import Dense, Flatten, Lambda, Activation, MaxPooling2D
from keras.layers.convolutional import Convolution2D
from keras.models import Sequential
from keras.optimizers import Adam

# a module helps pre-processing
import preProcess
MODEL_NAME = 'model.json'
MODEL_WEIGHT = 'model.h5'

tf.python.control_flow_ops = tf

number_of_epochs = 8
number_of_samples_per_epoch = 20032
number_of_validation_samples = 6400
learning_rate = 1e-4
activation_relu = 'relu'

# the model is based on NVIDIA's "End to End Learning for Self-Driving Cars" paper
# from https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
from keras.layers import BatchNormalization
model = Sequential()

# normalization layer
model.add(BatchNormalization(epsilon=0.001, mode=2, axis=1,input_shape=(64,64,3)))
# conv 1: output 24, kernel size 5x5
model.add(Convolution2D(24, 5,5, border_mode='valid', activation='relu', subsample=(2,2)))
# conv 2: output 36, kernel size 5x5
model.add(Convolution2D(36, 5,5, border_mode='valid', activation='relu', subsample=(2,2)))
# conv 3: output 48, kernel size 5x5
model.add(Convolution2D(48, 5,5, border_mode='same', subsample=(2, 2)))
model.add(Activation(activation_relu))d
# maxpooling
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
# conv 4: output 64, kernel size 3x3
model.add(Convolution2D(64, 3,3, border_mode='valid', activation='relu', subsample=(1,1)))
# conv 5: output 64, kernel size 3x3
model.add(Convolution2D(64, 3,3, border_mode='valid', activation='relu', subsample=(1,1)))
# flatten
model.add(Flatten())
# 4 fully-connected layers :
model.add(Dense(1164, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
# to output
model.add(Dense(1, activation='tanh'))

model.summary()
#using adam
model.compile(optimizer=Adam(learning_rate), loss="mse")

# create two batch generators for training and validation
train_gen = preProcess.imgBatchGenerator()
validation_gen = preProcess.imgBatchGenerator()

history = model.fit_generator(train_gen,
                              samples_per_epoch=number_of_samples_per_epoch,
                              nb_epoch=number_of_epochs,
                              validation_data=validation_gen,
                              nb_val_samples=number_of_validation_samples,
                              verbose=1)

# finally save our model and weights
# save arch in .json and weights in .h5
import json

json_string = model.to_json()
with open(MODEL_NAME, 'w') as outfile:
    json.dump(json_string, outfile)
model.save_weights(MODEL_WEIGHT)

# save arch in .png
from keras.utils.visualize_util import plot
plot(model, to_file='./examples/model.png')
