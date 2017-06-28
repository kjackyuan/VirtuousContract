import cv2
import imutils
import numpy as np


def get_hsv(window_name):
    MinH = cv2.getTrackbarPos('MinH', window_name)
    MaxH = cv2.getTrackbarPos('MaxH', window_name)
    MinS = cv2.getTrackbarPos('MinS', window_name)
    MaxS = cv2.getTrackbarPos('MaxS', window_name)
    MinV = cv2.getTrackbarPos('MinV', window_name)
    MaxV = cv2.getTrackbarPos('MaxV', window_name)
    return [MinH, MaxH, MinS, MaxS, MinV, MaxV]


def standard_recording(cap):
    ret, img = cap.read()
    img = cv2.resize(img, (300, 200))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return img, hsv


class Calibrator(object):
    # (x, y)
    centers = {
        0: (-1, -1),
        1: (-1, -1),
        2: (-1, -1),
        3: (-1, -1)
    }

    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype='float32')

        # the top-left point will have the smallest sum, whereas
        # the bottom-right point will have the largest sum
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        # top-right point will have the smallest difference,
        # whereas the bottom-left will have the largest difference
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        # return the ordered coordinates
        return rect

    def find_centers(self, img):
        '''
        :param img: binary image
        :return:
        '''
        countours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        countours = countours[0] if imutils.is_cv2() else countours[1]
        countours = sorted(countours, key=lambda x: cv2.contourArea(x), reverse=True)[:4]

        countour_center = []
        for c in countours:
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"]) if M["m00"] > 0 else 0
            cY = int(M["m01"] / M["m00"]) if M["m00"] > 0 else 0
            countour_center.append(np.array([cX, cY]))

        countour_center = np.array(countour_center)
        countour_center = self.order_points(countour_center)

        for idx, center in enumerate(countour_center):
            self.centers[idx] = center

    def transform_img(self, img):
        centers = np.array([self.centers[0], self.centers[1], self.centers[2], self.centers[3]])
        (tl, tr, br, bl) = centers

        # compute the width of the new image, which will be the
        # maximum distance between bottom-right and bottom-left
        # x-coordiates or the top-right and top-left x-coordinates
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        # compute the height of the new image, which will be the
        # maximum distance between the top-right and bottom-right
        # y-coordinates or the top-left and bottom-left y-coordinates
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        # now that we have the dimensions of the new image, construct
        # the set of destination points to obtain a "birds eye view",
        # (i.e. top-down view) of the image, again specifying points
        # in the top-left, top-right, bottom-right, and bottom-left
        # order
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        # compute the perspective transform matrix and then apply
        M = cv2.getPerspectiveTransform(centers, dst)
        warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

        # return the warped image
        return warped

    def calibrate_calibrator(self, cap_stream):
        windowName_calibration = 'positional calibration'
        cv2.namedWindow(windowName_calibration)
        cv2.createTrackbar('MinH', windowName_calibration, 0, 255, lambda x: x)
        cv2.createTrackbar('MaxH', windowName_calibration, 240, 255, lambda x: x)
        cv2.createTrackbar('MinS', windowName_calibration, 65, 255, lambda x: x)
        cv2.createTrackbar('MaxS', windowName_calibration, 255, 255, lambda x: x)
        cv2.createTrackbar('MinV', windowName_calibration, 144, 255, lambda x: x)
        cv2.createTrackbar('MaxV', windowName_calibration, 255, 255, lambda x: x)
        cv2.createTrackbar('Median', windowName_calibration, 0, 100, lambda x: x)

        windowName_median_filter = 'median filter xcaliber'
        cv2.namedWindow(windowName_median_filter)
        cv2.createTrackbar('ksize', windowName_median_filter, 5, 100, lambda x: x)

        calibrated = False
        while True:
            img, hsv = standard_recording(cap_stream)
            MinH, MaxH, MinS, MaxS, MinV, MaxV = get_hsv(windowName_calibration)
            hsv_filter = cv2.inRange(hsv, (MinH, MinS, MinV), (MaxH, MaxS, MaxV))

            ksize = cv2.getTrackbarPos('ksize', windowName_median_filter)
            if ksize % 2 == 0:
                ksize += 1

            hsv_filter = cv2.dilate(hsv_filter, (5, 5), iterations=5)
            hsv_filter = cv2.medianBlur(hsv_filter, ksize)

            cv2.imshow('img', img)
            cv2.imshow('hsv', hsv_filter)
            if calibrated:
                cv2.imshow('transformed', self.transform_img(img))

            k = cv2.waitKey(30)

            if k == ord('q'):
                self.find_centers(hsv_filter)
                calibrated = True
            elif k & 0xff == 27:
                cv2.destroyAllWindows()
                break
