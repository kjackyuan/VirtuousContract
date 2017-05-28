import cv2
import time
import os
import numpy as np

cap = cv2.VideoCapture(0)

num_square = 16
num_img = 1000

root_dir = 'image_data'
main_dir = 'realtime_raw_img_16'

main_dir = os.path.join(root_dir, main_dir)

if not os.path.isdir(main_dir):
    os.mkdir(main_dir)


for pos in range(0, num_square):
    counter = 0
    record = False

    sub_dir = os.path.join(main_dir, str(pos))
    if not os.path.isdir(sub_dir):
        os.mkdir(sub_dir)

    print 'Read to record Position: %s' % pos

    while True:
        ret, img = cap.read()

        img = cv2.resize(img, (300, 200))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        cv2.imshow('img', img)
        cv2.imshow('gray', gray)

        if record:
            if counter/float(num_img) >= 0.75:
                print '%s: 75%% Done' % pos
            elif counter/float(num_img) >= 0.50:
                print '%s: 50%% Done' % pos
            elif counter / float(num_img) >= 0.25:
                print '%s: 25%% Done' % pos

            if counter >= num_img:
                break
            filename = os.path.join(sub_dir, '%s.png' % counter)
            with open(filename, 'wb') as f:
                cv2.imwrite(filename, gray)
            counter += 1

        k = cv2.waitKey(3)
        if k == ord('q'):
            record = True
            print 'Recording.... Position: %s' % pos
            time.sleep(3)
        elif k & 0xff == 27:
            break

cap.release()