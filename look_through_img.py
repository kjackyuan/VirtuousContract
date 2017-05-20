from utils import show_img, Dataset_64

a = Dataset_64('./data_64')

maxsize = len(a.training_img)

while True:
	id = int(raw_input("img id: "))
	if id >= maxsize:
		print 'id out of scope'
		continue
	show_img(a.training_img[id], 34, 50)

