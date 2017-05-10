from PIL import ImageFilter
import os
from PIL import Image
import sys
import numpy as np

def main():
	src = sys.argv[1]
	dest = sys.argv[2]

	if not os.path.exists(dest):
		os.makedirs(dest)

	for fn in os.listdir('./'+src):
		print(fn)
		fname = fn.split('.')[0]
		im = Image.open(os.path.join(src, fn)).convert('RGB')

		# im = crop(im)
		#if '.jpg' not in fn:
		im.save(os.path.join(dest, fname)+'.jpg', 'JPEG', quality=90)

def addBlur(im):
	w, h = im.size
	nim = im.resize((round(w*w/h),w), Image.ANTIALIAS)
	canvas = Image.new('RGB', (max(w,h), max(w,h)))  # e.g. ('RGB', (640, 480))

	nw, nh = nim.size
	rad = 14
	while nw > w:
		# print(nw, h, nh)
		# print(nw/nh)
		rad = rad - 1
		if(rad < 3):
			rad = 3
		nim = nim.resize((nw, nh), Image.ANTIALIAS)
		blurred = nim.filter(ImageFilter.GaussianBlur(radius=rad))
		canvas.paste(blurred, (round(0+w/2-nw/2),round(0+w/2-nh/2)))
		nw -= 5*w/h
		nh -= 5
		nw, nh = round(nw), round(nh)
	# nim = nim.resize((nw, nh), Image.ANTIALIAS)
	# black = black.filter(ImageFilter.GaussianBlur(radius=10))
	canvas.paste(im, (round(0+w/2-nw/2),round(0+w/2-nh/2)))
	return canvas

def crop(im):
	data = np.asarray(im)
	# print(data[:,0])
	data = cropLeft(data)
	data = cropRight(data)
	data = cropTop(data)
	data = cropBottom(data)

	im = Image.fromarray(data)
	w, h = im.size
	if w > h:
		black = addBlur(im)
	else:
		imRot = im.rotate(90, expand=True)
		black = addBlur(imRot)
		black = black.rotate(-90, expand=True)

	return black


def cropLeft(data):
	zas = True
	while zas:
		col = data[:, 0, 0]
		dif = abs(np.sum(col)/data.shape[0] - col[0])
		if dif > 5:
			zas = False
		else:
			data = data[:, 1:, :]
	return data


def cropRight(data):
	zas = True
	while zas:
		col = data[:, data.shape[1]-1, 0]
		dif = abs(np.sum(col)/data.shape[0] - col[0])
		if dif > 5:
			zas = False
		else:
			data = data[:, :data.shape[1]-1, :]
	return data

def cropTop(data):
	zas = True
	while zas:
		col = data[0, :, 0]
		dif = abs(np.sum(col)/data.shape[1] - col[0])
		if dif > 5:
			zas = False
		else:
			data = data[1:, :, :]
	return data


def cropBottom(data):
	zas = True
	while zas:
		col = data[data.shape[0]-1, :, 0]
		dif = abs(np.sum(col)/data.shape[1] - col[0])
		if dif > 5:
			zas = False
		else:
			data = data[:data.shape[0]-1, :, :]
	return data


if __name__ == '__main__':
	main()
