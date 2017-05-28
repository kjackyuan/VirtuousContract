import cv2
import time
import os
import numpy as np


def sobel_img(img):
    edges_x = cv2.Sobel(img, -1, 1, 0, 20)
    edges_y = cv2.Sobel(img, -1, 0, 1, 20)
    edges = edges_x + edges_y
    return edges

num_square = 16
num_img = 150

root_dir = 'image_data'
main_img_dir = 'realtime_raw_img_16'
main_img_dir = os.path.join(root_dir, main_img_dir)

out_img_dir = 'realtime_sobel_img_16'
out_img_dir = os.path.join(root_dir, out_img_dir)

if not os.path.isdir(out_img_dir):
    os.mkdir(out_img_dir)

for pos in range(0, num_square):
    sub_img_dir = os.path.join(main_img_dir, str(pos))
    sub_data_dir = os.path.join(out_img_dir, str(pos))

    if not os.path.isdir(sub_data_dir):
        os.mkdir(sub_data_dir)

    print pos

    for i in range(0, num_img):
        img_filepath = os.path.join(sub_img_dir, '%s.png' % i)
        out_img_filepath = os.path.join(sub_data_dir, '%s.png' % i)

        img = cv2.imread(img_filepath, 0)
        if img is None:
            continue
        edges = sobel_img(img)

        cv2.imwrite(out_img_filepath, edges)