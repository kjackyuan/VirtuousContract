import cv2
import time
import os
import numpy as np
from multiprocessing import Pool


num_square = 16
num_img = 1000

root_dir = 'image_data'
main_img_dir = 'delta_raw_img_16'
main_img_dir = os.path.join(root_dir, main_img_dir)

main_data_dir = 'delta_data_16'
main_data_dir = os.path.join(root_dir, main_data_dir)

if not os.path.isdir(main_data_dir):
	os.mkdir(main_data_dir)

normalize = lambda x: x/255.0
normalize = np.vectorize(normalize)

def convert_this_square(pos):
	sub_img_dir = os.path.join(main_img_dir, str(pos))
	sub_data_dir = os.path.join(main_data_dir, str(pos))

	if not os.path.isdir(sub_data_dir):
		os.mkdir(sub_data_dir)

	print pos

	for i in range(0, num_img):
		img_filepath = os.path.join(sub_img_dir, '%s.png' % i)
		data_filepath = os.path.join(sub_data_dir, '%s.csv' % i)

		img = cv2.imread(img_filepath, 0)
		img = img.reshape(200*300)
		img = normalize(img)

		img.tofile(data_filepath, sep=',', format='%10.10f')

pool = Pool(processes=4)  # start 4 worker processes
inputs = range(num_square)
pool.map(convert_this_square, inputs)


