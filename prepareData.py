# greyscale img
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import PIL.ImageOps
import os

path = 'data/train/imageRaw'
num_files = len([f for f in os.listdir(path)if os.path.isfile(os.path.join(path, f))])

def cropImages():
	imageFolder = 'data/train/imageRaw'
	labelFolder = 'data/train/labelRaw'

	for counter in range(num_files):
		image = Image.open(imageFolder + '/image' + str(counter) + '.png')
		croppedImage = image.crop((0, 0, 512, 512))
		croppedImage.save('data/train/imageCropped/' + str(counter) + '.png')

		label = Image.open(labelFolder + '/layer' + str(counter) + '.png')
		croppedLabel = label.crop((0, 0, 512, 512))
		croppedLabel.save('data/train/labelCropped/' + str(counter) + '.png')

def preprocessImage():
	folder = 'data/train/imageCropped'

	for image_name in os.listdir(folder):
		img = Image.open(folder + '/' + image_name).convert('L')
		img.save('data/train/image/' + image_name)


def preprocessLayer():
	folder = 'data/train/labelCropped'

	for image_path in os.listdir(folder):
		img = Image.open(folder + '/' + image_path)
		img = img.convert("RGBA")
		pixdata = img.load()

		for y in range(img.size[1]):
			for x in range(img.size[0]):
				if pixdata[x, y] == (0, 0, 0, 255):
					pixdata[x, y] = (0, 0, 0, 255)
				else:
					pixdata[x, y] = (255, 255, 255, 255)
		img.save('data/train/label/' + image_path,'PNG')

if __name__ == "__main__":

	cropImages()
	preprocessImage()
	preprocessLayer()
