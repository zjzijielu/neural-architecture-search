from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.models import Sequential, Model 
from keras.layers import Conv2D, GlobalAveragePooling2D, Input, Dense
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical

6 # prepare the training data for the NetworkManager
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

dataset = [x_train, y_train, x_test, y_test]

batch_size = 128
epochs = 20

filter_1 = 64
filter_2 = 64
filter_3 = 32
filter_4 = 64

kernel_1 = 3
kernel_2 = 3
kernel_3 = 3
kernel_4 = 3

ip = Input(shape=(32, 32, 3))

x = Conv2D(filter_1, (kernel_1, kernel_1), strides=(2, 2), padding='same', activation='relu')(ip)
x = Conv2D(filter_2, (kernel_2, kernel_2), strides=(1, 1), padding='same', activation='relu')(x)
x = Conv2D(filter_3, (kernel_3, kernel_3), strides=(2, 2), padding='same', activation='relu')(x)
x = Conv2D(filter_4, (kernel_4, kernel_4), strides=(1, 1), padding='same', activation='relu')(x)
x = GlobalAveragePooling2D()(x)
x = Dense(10, activation='softmax')(x)

model = Model(ip, x)

model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])

earlystopper = EarlyStopping(monitor='val_acc', min_delta=0.01, patience=2, verbose=1)

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test),
          callbacks=[earlystopper])


