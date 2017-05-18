from PIL import ImageFilter
import os, argparse
from PIL import Image
import sys
import numpy as np

def main(src, dest, m, c, w, h):
	# src = sys.argv[1]
	# dest = sys.argv[2]
	for fn in os.listdir(src):
		print(fn)
		fname = fn.split('.')[0]
		im = Image.open(os.path.join(src, fn)).convert('RGB')

		if m:
			im = crop_and_blur(im)
		elif c:
			im = crop(im)
		if w and h:
			im = im.resize((w,h), Image.ANTIALIAS)
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

def crop_and_blur(im):
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


def crop(im):
	data = np.asarray(im)
	# print(data[:,0])
	data = cropLeft(data)
	data = cropRight(data)
	data = cropTop(data)
	data = cropBottom(data)

	im = Image.fromarray(data)

	return im


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
	parser = argparse.ArgumentParser("Converts images to jpg.")
	parser.add_argument("imagesFolder", type=str, help="input_folder output_folder", nargs=2)

	parser.add_argument("--m", action="store_true", help="masked - smart images preprocessing")
	parser.add_argument("--c", action="store_true", help="crop - removes padding")

	parser.add_argument("-wh", help="set output size (w x h)", type=int, nargs=2)

	args = parser.parse_args()	

	if not os.path.exists(os.path.abspath(args.imagesFolder[0])): raise OSError(2, 'No such file or directory', os.path.abspath(args.imagesFolder[0]))
	if not os.path.exists(os.path.abspath(args.imagesFolder[1])): os.makedirs(os.path.abspath(args.imagesFolder[1]))

	if args.c: 	
		print("Method: crop")
		C = True
	else:
		C = False
	if args.m: 	
		print("Method: mask")
		M = True
	else:
		M = False
	if args.wh: 
		w, h = args.wh[0], args.wh[1]	
	else:
		w, h = 0, 0
	main(args.imagesFolder[0], args.imagesFolder[1], M, C, w, h)
