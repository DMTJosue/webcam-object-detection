import cv2
import numpy as np 

cap = cv2.VideoCapture(0)

fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()

    if not ret:
        break
    frame=cv2.resize(frame, (320, 240))

    fgmask = fgbg.apply(frame)
    fgmask =cv2.resize(fgmask, (320, 240))

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0, 120, 70])  
    upper_red = np.array([10, 255, 255])  

    
    mask = cv2.inRange(hsv, lower_red, upper_red)

    
    res = cv2.bitwise_and(frame, frame, mask=mask)
    res =cv2.resize(res, (320, 240))

    cv2.imshow('Original', frame)  
    cv2.imshow('Mouvement detecté', fgmask) 
    cv2.imshow('Objet detecté (rouge)', res)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()