import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Reshape, Conv2D, Flatten,MaxPooling2D
from keras.optimizers import adam

file = open("train.csv")
data_train = pd.read_csv(file)
y_train = np.array(data_train.iloc[:, 0])
x_train = np.array(data_train.iloc[:, 1:])

file=open("test.csv")
data_test=pd.read_csv(file)
x_test=np.array(data_test)

def output_prediction(y_p):
    data_predict = {"ImageId":range(1, 28001), "Label":y_p}
    data_predict = pd.DataFrame(data_predict)
    data_predict.to_csv("output.csv", index = False)

model = Sequential()
y_train = keras.utils.to_categorical(y_train, num_classes=10)

model.add(Reshape(target_shape=(1, 28, 28), input_shape=(784,)))
model.add(Conv2D(kernel_size=(5, 5), filters=32, padding="same", activation ='relu',kernel_initializer="uniform"))
model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))
model.add(Dropout(0.25))
model.add(Conv2D(kernel_size=(5, 5), filters=32, padding="same", activation ='relu',kernel_initializer="uniform"))
model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(output_dim=1000, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(output_dim=100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(output_dim=10, activation='softmax'))

adam = keras.optimizers.Adam(lr=0.0005)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

model.fit(x_train, y_train, epochs=300, batch_size=50)
y_p = model.predict_classes(x_test)

output_prediction(y_p)