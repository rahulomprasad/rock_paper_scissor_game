

import cv2
import os
import sys


label_name="rock..."
IMG_SAVE_PATH = 'image_data'
IMG_CLASS_PATH = os.path.join(IMG_SAVE_PATH, label_name)

cap = cv2.VideoCapture(0)

start = False
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    if count == 200:
        break

    cv2.rectangle(frame, (100, 100), (400, 400), (255, 255, 255), 2)

    if start:
        roi = frame[100:500, 100:500]
        save_path = os.path.join(IMG_CLASS_PATH, '{}.jpg'.format(count + 1))
        cv2.imwrite(save_path, roi)
        count += 1
        start= not start

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "Collecting {}".format(count),
            (5, 50), font, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Collecting images", frame)

    k = cv2.waitKey(0)
    if k %256==32:
        start = not start

    if k %256==27:
        break

print("\n{} image(s) saved to {}".format(count, IMG_CLASS_PATH))
cap.release()
cv2.destroyAllWindows()
