'''
D’(x,y,t) = |I(x,y,t)-I(x,y,t-1)|
Video code cv2 template: https://pythonprogramming.net/loading-video-python-opencv-tutorial/
'''

import cv2
import numpy as np

cap = cv2.VideoCapture(0)
prev_frame = None
template = cv2.imread('img/selfie1_scissor_hand.png',0)

found = []
w, h = template.shape[::-1]

while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #converts captured frame to Grayscale for easier analysis
    
    res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    print(res)
    threshold = 0.8
    loc = np.where( res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

    cv2.imshow('frame', frame)
    
    found.append(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): #Press q to quit video capture
        print(str(len(found)), "found")
        break

cap.release()
cv2.destroyAllWindows()

