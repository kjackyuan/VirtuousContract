import cv2
import time
import os
from tqdm import tqdm

from img_utils import create_dir_if_dne
from img_utils import get_hsv, Calibrator, standard_recording

cap = cv2.VideoCapture(0)


# Calibration:
caliber = Calibrator()
caliber.calibrate_calibrator(cap)


# recording configurations
num_frame = 20
num_region = 9
num_position = 5
num_iteration = 6

region_fov = -1
region_coordinate = {
    0: (),
    1: (),
    2: (),
    3: (),
    4: (),
    5: (),
    6: (),
    7: (),
    8: ()
}

root_dir = 'image_data'
main_dir = 'double_tap'

main_dir = os.path.join(root_dir, main_dir)

create_dir_if_dne(main_dir)


# control window
windowName_hand = 'hand_filter'
cv2.namedWindow(windowName_hand)
cv2.createTrackbar('MinH', windowName_hand, 0, 255, lambda x: x)
cv2.createTrackbar('MaxH', windowName_hand, 74, 255, lambda x: x)
cv2.createTrackbar('MinS', windowName_hand, 69, 255, lambda x: x)
cv2.createTrackbar('MaxS', windowName_hand, 255, 255, lambda x: x)
cv2.createTrackbar('MinV', windowName_hand, 0, 255, lambda x: x)
cv2.createTrackbar('MaxV', windowName_hand, 255, 255, lambda x: x)

windowName_median_filter = 'median_filter'
cv2.namedWindow(windowName_median_filter)
cv2.createTrackbar('ksize', windowName_median_filter, 13, 100, lambda x: x)


def record_and_filter():
    img, hsv = standard_recording(cap)
    cv2.imshow('original', img)

    img = caliber.transform_img(img)
    hsv = caliber.transform_img(hsv)

    MinH, MaxH, MinS, MaxS, MinV, MaxV = get_hsv(windowName_hand)
    hsv_filter = cv2.inRange(hsv, (MinH, MinS, MinV), (MaxH, MaxS, MaxV))

    cv2.imshow('img', img)
    cv2.imshow('hsv', hsv)

    ksize = cv2.getTrackbarPos('ksize', windowName_median_filter)
    if ksize % 2 == 0:
        ksize += 1

    hsv_filter = cv2.dilate(hsv_filter, (5, 5), iterations=5)
    hsv_filter = cv2.medianBlur(hsv_filter, ksize)
    cv2.imshow('hsv_filter', hsv_filter)
    k = cv2.waitKey(1)

    return hsv_filter, k


# recording
for region in range(0, num_region):
    for pos in range(0, num_position):
        for iter in range(0, num_iteration):
            sub_dir = os.path.join(main_dir, '%s_%s_%s' % (region, pos, iter))
            create_dir_if_dne(sub_dir)

            print '\n'
            print 'Previewing... Region: %s, Position: %s, Iteration: %s' % (region, pos, iter)
            print 'Press Q to Begin'

            while True:
                _, k = record_and_filter()

                if k == ord('q'):
                    break
                elif k & 0xff == 27:
                    break

            for id in tqdm(range(num_frame)):
                hsv_filter, _ = record_and_filter()

                filename = os.path.join(sub_dir, '%s.png' % id)
                with open(filename, 'wb') as f:
                    cv2.imwrite(filename, hsv_filter)


cap.release()