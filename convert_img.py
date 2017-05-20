import cv2
import time
import os
import numpy as np

normalize = lambda x: x/255.0
normalize = np.vectorize(normalize)


for pos in range(0, 64):
	os.mkdir('data_64/%s' % pos)
	
	print pos
	for i in range(1,502):
		filename = 'training_data_64/%s/%s.png' % (pos, i)

		img = cv2.imread(filename, 0)
		img = img.reshape(200*300)
		img = normalize(img)

		output_filename = 'data_64/%s/%s.csv' % (pos, i)
		img.tofile(output_filename, sep=',', format='%10.10f')