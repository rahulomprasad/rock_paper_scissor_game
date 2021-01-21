


import cv2
import os

label_name="scissors"
IMG_SAVE_PATH = 'image'
IMG_CLASS_PATH = os.path.join(IMG_SAVE_PATH, label_name)

cap = cv2.VideoCapture(0)

start = False
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    if count == 2000:
        break

    cv2.rectangle(frame, (100, 100), (800, 500), (255, 255, 255), 2)
    roi = frame[100:500, 100:800]

    if start:
        
        save_path = os.path.join(IMG_CLASS_PATH, '{}.jpg'.format(count + 1))
        cv2.imwrite(save_path, roi)
        count += 1
        start=not start

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
