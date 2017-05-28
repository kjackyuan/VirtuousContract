import cv2
import os

img_dir = 'image_data/delta_raw_img_16'
img_dir2 = 'image_data/realtime_raw_img_16'
a = -1
b = -1

def get_ab():
    a = float(raw_input('a: '))
    b = float(raw_input('b: '))
    return (a, b)

while True:
    for pos in range(0, 4):
        for id in range(0, 200):
            # if a == -1 or b == -1:
            #     a, b = get_ab()

            img = cv2.imread(os.path.join(img_dir, '%s/%s.png' % (pos, id)), 0)
            edges_x = cv2.Sobel(img, -1, 1, 0, 20)
            edges_y = cv2.Sobel(img, -1, 0, 1, 20)
            edges = edges_x + edges_y

            cv2.imshow('1', img)
            cv2.imshow('x', edges_x)
            cv2.imshow('y', edges_y)
            cv2.imshow('all', edges)
            #
            # img = cv2.imread(os.path.join(img_dir2, '%s/%s.png' % (pos, id)), 0)
            # if img is None:
            #     continue
            #
            # edges_x = cv2.Sobel(img, -1, 1, 0, 20)
            # edges_y = cv2.Sobel(img, -1, 0, 1, 20)
            # edges = edges_x + edges_y
            #
            # cv2.imshow('2', img)
            # cv2.imshow('x_2', edges_x)
            # cv2.imshow('y_2', edges_y)
            # cv2.imshow('all_2', edges)


            k = cv2.waitKey(30)
            if k == ord('q'):
                a, b = get_ab()