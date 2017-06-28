from multiprocessing import Pool

import numpy as np
import cv2
import os


root_dir = 'image_data'
main_img_dir = 'beta_raw_img_16'
main_img_dir = os.path.join(root_dir, main_img_dir)

windowName = 'window'
cv2.namedWindow(windowName)
cv2.createTrackbar('MinH', windowName, 0, 180, lambda x: x)
cv2.createTrackbar('MaxH', windowName, 0, 180, lambda x: x)
cv2.createTrackbar('MinS', windowName, 0, 255, lambda x: x)
cv2.createTrackbar('MaxS', windowName, 0, 255, lambda x: x)
cv2.createTrackbar('MinV', windowName, 0, 255, lambda x: x)
cv2.createTrackbar('MaxV', windowName, 0, 255, lambda x: x)
cv2.createTrackbar('MinT', windowName, 0, 255, lambda x: x)
cv2.createTrackbar('MaxT', windowName, 0, 255, lambda x: x)

slope_1 = 2.64151
slope_2 = 2.75

def crop_img(args):
    row, col_data = args
    for col in range(len(col_data)):
        if col <= 53 and row <= int((53 - col) * slope_1):
            col_data[col] = [0, 0, 0]
        elif col >= 260 and row <= int((col - 260) * slope_2):
            col_data[col] = [0, 0, 0]
    return (row, col_data)

pool = Pool(processes=4)  # start 4 worker processes

for pos in range(16):
    for id in range(0, 600):
        img = cv2.imread(os.path.join(main_img_dir, '%s' % pos, '%s.png' % id))
        img = img[0:180, 0:300]

        inputs = []
        for row in range(len(img)):
            inputs.append([row, img[row]])

        results = pool.map(crop_img, inputs)

        results = sorted(results, key=lambda x: x[0])
        results = [_[1] for _ in results]

        img = np.array(results)
        cv2.imshow('a', img)

        # MinH = cv2.getTrackbarPos('MinH', windowName)
        # MaxH = cv2.getTrackbarPos('MaxH', windowName)
        # MinS = cv2.getTrackbarPos('MinS', windowName)
        # MaxS = cv2.getTrackbarPos('MaxS', windowName)
        # MinV = cv2.getTrackbarPos('MinV', windowName)
        # MaxV = cv2.getTrackbarPos('MaxV', windowName)
        # MinT = cv2.getTrackbarPos('MinT', windowName)
        # MaxT = cv2.getTrackbarPos('MaxT', windowName)
        MinT = 40
        MaxT = 75
        #f_img = cv2.inRange(img, (MinT, MinT, MinT), (MaxT, MaxT, MaxT))
        #f_img = cv2.GaussianBlur(f_img, (3, 3), 0)
        f_img = cv2.Sobel(img,cv2.CV_8U,1,0,ksize=3)

        cv2.imshow('b', f_img)

        cv2.waitKey(30)