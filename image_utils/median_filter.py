import cv2


cap = cv2.VideoCapture(0)

windowName_median_filter = 'median_filter'
cv2.namedWindow(windowName_median_filter)
cv2.createTrackbar('ksize', windowName_median_filter, 13, 100, lambda x: x)
cv2.createTrackbar('mult', windowName_median_filter, 1, 30, lambda x: x)


windowName = 'window'
cv2.namedWindow(windowName)
cv2.createTrackbar('MinH', windowName, 0, 255, lambda x: x)
cv2.createTrackbar('MaxH', windowName, 74, 255, lambda x: x)
cv2.createTrackbar('MinS', windowName, 69, 255, lambda x: x)
cv2.createTrackbar('MaxS', windowName, 255, 255, lambda x: x)
cv2.createTrackbar('MinV', windowName, 0, 255, lambda x: x)
cv2.createTrackbar('MaxV', windowName, 255, 255, lambda x: x)

while True:
    ret, img = cap.read()
    img = cv2.resize(img, (300, 200))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    MinH = cv2.getTrackbarPos('MinH', windowName)
    MaxH = cv2.getTrackbarPos('MaxH', windowName)
    MinS = cv2.getTrackbarPos('MinS', windowName)
    MaxS = cv2.getTrackbarPos('MaxS', windowName)
    MinV = cv2.getTrackbarPos('MinV', windowName)
    MaxV = cv2.getTrackbarPos('MaxV', windowName)
    hsv_filter = cv2.inRange(hsv, (MinH, MinS, MinV), (MaxH, MaxS, MaxV))
    hsv_filter = cv2.dilate(hsv_filter, (5, 5), iterations=5)

    ksize = cv2.getTrackbarPos('ksize', windowName_2b)
    mult = cv2.getTrackbarPos('mult', windowName_2b)

    if ksize % 2 == 0:
        ksize += 1

    for _ in range(mult):
        hsv_filter = cv2.medianBlur(hsv_filter, ksize)

    cv2.imshow('img', img)
    cv2.imshow('hsv_filter', hsv_filter)

    k = cv2.waitKey(30)