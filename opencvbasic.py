import cv2
import time
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

counter = 0
record = False

while True:
    ret, img = cap.read()

    img = cv2.resize(img, (300, 200))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    #
    # print faces
    # for (x,y,w,h) in faces:
    #     cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
    #     roi_gray = gray[y:y+h, x:x+w]
    #     roi_color = img[y:y+h, x:x+w]
    #     eyes = eye_cascade.detectMultiScale(roi_gray)
    #     for (ex, ey, ew, eh) in eyes:
    #         cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0,255,0), 2)


    cv2.imshow('img', img)

    if record:
        if counter == 0:
            time.sleep(3)
        elif counter > 300:
            break
        result = []
        counter += 1
        for i in gray:
            for r in i:
                result.append(r/255.0)
        filename = 'training_data/bot_right/%s.csv' % counter
        np.asarray(result).tofile(filename,sep=',',format='%10.10f')

    k = cv2.waitKey(30)
    if k == ord('q'):
        record = True
    elif k & 0xff == 27:
        break


cap.release()