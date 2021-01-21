from keras.models import load_model
import cv2
import numpy as np
from random import choice

REV_CLASS_MAP = {
    0: "rock",
    1: "paper",
    2: "scissors",
    3: "none"
}


def mapper(val):
    return REV_CLASS_MAP[val]


def calculate_winner(move1, move2):
    if move1 == move2:
        return "Tie"

    if move1 == "rock":
        if move2 == "scissors":
            return "User"
        if move2 == "paper":
            return "Computer"

    if move1 == "paper":
        if move2 == "rock":
            return "User"
        if move2 == "scissors":
            return "Computer"

    if move1 == "scissors":
        if move2 == "paper":
            return "User"
        if move2 == "rock":
            return "Computer"
    else:
        return "no"


model = load_model("rock-paper-scissors-model.h5")

cap = cv2.VideoCapture(0)
#cam = cv2.VideoCapture(1)


start=False
count=0
while True:


    ret, frame = cap.read()
    
    if not ret:
        continue

    cv2.rectangle(frame, (100, 100), (800, 500), (255, 255, 255), 2)

    roi = frame[100:500, 100:800]

    if start:
        
        
        img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (28, 28))
        img=img[:,:,0]
        img=np.asarray(img)
        img=img.reshape(1,28,28,1)
        pred = model.predict(img)
        move_code = np.argmax(pred[0])
        user_move = mapper(move_code)
        
        if user_move != "none":
            computer_move = choice(['rock', 'paper', 'scissors'])
            winner = calculate_winner(user_move, computer_move)
            if winner == "User":
                count+=1
            elif winner == "Computer":
                count=count-1
        else:
            computer_move = "none"
            winner = "Waiting..."
            

        print(user_move, computer_move, winner)
        
        start=not start


        # display the information
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "Your Move: " + user_move,
                    (50, 50), font, 1.2, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "Computer's Move: " + computer_move,
                    (10, 240), font, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "Winner: " + winner,
                    (120, 160), font, 2, (0, 0, 255), 4, cv2.LINE_AA)

        
        #print(1111)
        
        

    #if computer_move_name != "none":
        icon = cv2.imread(
            "computer/{}.jpg".format(computer_move))
        icon = cv2.resize(icon, (200, 200))
        #icon = icon[:,:,0]
        frame[280:480, 0:200] = icon

     

    cv2.imshow("USER", frame)

    k = cv2.waitKey(0)
    if k%256==27:
        break
    if k%256==32:
        start=not start
    
    
    
if count>0:   
    print("Congratulations you have won")
elif count<0:
    print("you lost the game")
else:
    "DRAW"

cap.release()
cv2.destroyAllWindows()
