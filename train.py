import cv2

import numpy as np

from keras.optimizers import Adam

from keras.utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense,Conv2D,Flatten,MaxPooling2D

import os

IMG_SAVE_PATH = 'image'

CLASS_MAP = {
    "rock": 0,
    "paper": 1,
    "scissors": 2,
    "none": 3
}

NUM_CLASSES = len(CLASS_MAP)


def mapper(val):
    return CLASS_MAP[val]


def get_model():
    
    CNN_model=Sequential()
    #Performing Convolution
    CNN_model.add(Conv2D(64,kernel_size=3,activation='relu',input_shape=(28,28,1)))
    #Performing pooling
    CNN_model.add(MaxPooling2D(pool_size=(2,2)))
    #Performing Convolution
    CNN_model.add(Conv2D(64,kernel_size=3,activation='relu'))
    #Performin pooling 
    CNN_model.add(MaxPooling2D(pool_size=(2,2)))
    #Performing Convolution
    CNN_model.add(Conv2D(64,kernel_size=3,activation='relu',input_shape=(28,28,1)))
    #Performing pooling
    CNN_model.add(MaxPooling2D(pool_size=(2,2)))
    #Performing Flattening
    CNN_model.add(Flatten())
    #Adding numbers of nodes in hidden 
    CNN_model.add(Dense(4,activation='softmax'))
    return CNN_model


# load images from the directory
dataset = []
for directory in os.listdir(IMG_SAVE_PATH):
    path = os.path.join(IMG_SAVE_PATH, directory)
    if not os.path.isdir(path):
        continue
    for item in os.listdir(path):
        # to make sure no hidden files get in our way
        if item.startswith("."):
            continue
        img = cv2.imread(os.path.join(path, item))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (28, 28))
        img = img[:,:,0]
        dataset.append([img, directory])


#preprocessing data
data, labels = zip(*dataset)
labels = list(map(mapper, labels))
data = np.asarray(data)
#reshape to specific format
data=data.reshape(8000,28,28,1)
labels= to_categorical(labels)


# define the model
model = get_model()
model.compile(
    optimizer=Adam(lr=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# start training
model.fit(np.array(data), np.array(labels), epochs=25)

# save the model for later use
model.save("rock-paper-scissors-model.h5")
