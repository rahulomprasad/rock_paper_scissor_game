from keras.models import load_model
import cv2
import numpy as np
import sys

#filepath = sys.argv[1]

REV_CLASS_MAP = {
    0: "rock",
    1: "paper",
    2: "scissors",
    3: "none"
}


def mapper(val):
    return REV_CLASS_MAP[val]


model = load_model("rock-paper-scissors-model.h5")

# prepare the image
img = cv2.imread('test/38.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (28, 28))
img = img[:,:,0]
img=np.asarray(img)
img=img.reshape(1,28,28,1)

# predict the move made
#pred = model.predict(np.array([img]))
pred = model.predict(img)
move_code = np.argmax(pred[0])
move_name = mapper(move_code)

print("Predicted: {}".format(move_name))
