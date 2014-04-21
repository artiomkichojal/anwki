#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import numpy as np
import cv2
import math as m



def imgToGray(filename):
	image = cv2.imread(filename+'.jpg')
	# print "processing picture to gray"
	# cv2.imwrite(filename + '_g.jpg',np.mean(image,axis=2))
	# print "to Gray ready"
	return np.mean(image,axis=2)
"""
binarisierungsfunktion nach niblack
"""
def niblack(filename):
	# image = imgToGray(filename) #make gray picture
	# image = cv2.imread(filename + '_g.jpg')# graues bild einlesen	
	image = imgToGray(filename)
	print "processing picture niblack..."
		
	fn_w = 0
	fn_h = 0
	counter = 0
	while (fn_h+1)*15 < image.shape[1]:	
		while (fn_w+1)*15 < image.shape[0]:	
			
			x0 = fn_w*15
			x1 = (fn_w+1)*15
			y0 = fn_h*15
			y1 = (fn_h+1)*15
			
			#15x15 pixel großes fenster
			fenster = image[x0:x1,y0:y1] 
			treshold = np.mean(fenster) - 0.2*np.std(fenster) #treshold nach niblack
			
			#faerbe jeden pixel in schwarz oder weiss um
			fenster[fenster<treshold] = 0
			fenster[fenster>treshold] = 255
			fn_w += 1
		fn_w = 0
		fn_h += 1
		
	cv2.imwrite(filename + '_nib.jpg',image) #in bild schreiben
	print "to niblack ready"

"""
binarisierungsfunktion nach sauvola
"""
def sauvola(filename):
	#imgToGray(filename) #make gray picture
	# image = cv2.imread(filename + '.jpg')# graues bild  einlesen	
	# image = np.mean(image,axis = 2)
	image = imgToGray(filename)
	print "processing picture sauvola..."	
	fn_w = 0
	fn_h = 0

	while (fn_h+1)*15 < image.shape[1]:	
		while (fn_w+1)*15 < image.shape[0]:	
			
			x0 = fn_w*15
			x1 = (fn_w+1)*15
			y0 = fn_h*15
			y1 = (fn_h+1)*15
			
			#15x15 pixel großes fenster
			fenster = image[x0:x1,y0:y1] 
			treshold = np.mean(fenster) *(1+0.2*(np.std(fenster)/128 - 1)) #treshold nach sauvola
			
			#faerbe jeden pixel in schwarz oder weiss um
			fenster[fenster<treshold] = 0
			fenster[fenster>treshold] = 255
			fn_w += 1
		fn_w = 0
		fn_h += 1
		
	cv2.imwrite(filename + '_sauv.jpg',image) #in bild schreiben
	print "to sauvola ready"	

"""
binarisierungsfunktion nach nick
"""
def nick_my(filename):
	# imgToGray(filename) #make gray picture
	# image = cv2.imread(filename + '_g.jpg')# graues bild  einlesen
	image = imgToGray(filename)	
	print "processing picture nick..."	
	fn_w = 0
	fn_h = 0
	while (fn_h+1)*15 < image.shape[1]:	
		while (fn_w+1)*15 < image.shape[0]:	
			
			x0 = fn_w*15
			x1 = (fn_w+1)*15
			y0 = fn_h*15
			y1 = (fn_h+1)*15
			
			#15x15 pixel großes fenster
			fenster = image[x0:x1,y0:y1] 
			m = np.mean(fenster)
			b = np.sum(fenster**2) - m**2
			wurzel = np.sqrt(b/(15*15))
			treshold = m - 0.1 * wurzel  #treshold nach nick
			
			
			#faerbe jeden pixel in schwarz oder weiss um
			fenster[fenster<treshold] = 0
			fenster[fenster>treshold] = 255
			fn_w += 1
		fn_w = 0
		fn_h += 1
	cv2.imwrite(filename + '_nick.jpg',image) #in bild schreiben
	print "to nick ready"

def nick(filename):
	image = cv2.imread(filename + '.jpg')
	cv2.resize(image,(0,0),fx = 0.5,fy=05)
	height, width, depth = image.shape

	image3 = np.mean(image, axis = 2) #graues bild

	for i in range(height/15):
		for j in range(width/15):
			
			fenster = image3[i*15:i*15+15,j*15:j*15+15]
			m = np.mean(fenster)
			b = np.sum(fenster**2) - m**2
			wurzel = np.sqrt(b/(15*15))
			treshhold = m - 0.1*wurzel
			
			fenster[fenster<treshhold] = 0
			fenster[fenster>treshhold] = 255
	cv2.imwrite(filename + '_nick.jpg',image3) #in bild schreiben
	print "to nick ready"
	
"""
create image with resolution width x height
"""
def resizeIm(filename,x_dim):
	oriimage = cv2.imread(filename + '.jpg')
	if x_dim < oriimage.shape[1]:
		faktor = float(float(x_dim)/float(oriimage.shape[1]))
		newimage = cv2.resize(oriimage,(0,0),fx=faktor, fy=faktor)
		cv2.imwrite(filename + '_res.jpg',newimage)
		return True
	return False
#resizeIm("34", 1024, 800)
#nick("01")

def integralBild(data_image):
	return np.cumsum(np.cumsum(data_image,axis=0),axis=1)

def niblack2(filename):
	if resizeIm(filename, 600):
		gray_im = imgToGray(filename+"_res")
	else:
		gray_im = imgToGray(filename)
	cum_sum = integralBild(gray_im)
	newimage = np.empty((gray_im.shape[0],gray_im.shape[1]))
	for i in range(8,gray_im.shape[0] - 7):
		for j in range(8,gray_im.shape[1] - 7):
			#print gray_im[i-7:i+7,j-7:j+7]
			mean = (cum_sum[i-7][j-7] + cum_sum[i+7][j+7]  - cum_sum[i+7][j-7] - cum_sum[i-7][j+7])/(14*14)
			std = np.std(gray_im[i-7:i+7,j-7:j+7])
			treshhold = mean - 0.2*std
			#print "mean: ", mean, " std: ",std
			if gray_im[i][j] > treshhold:
				newimage[i][j] = 255
			else :
				newimage[i][j] = 0
	cv2.imwrite(filename + '_nib2.jpg',newimage)
def sauvola2(filename):
	if resizeIm(filename, 800):
		gray_im = imgToGray(filename+"_res")
	else:
		gray_im = imgToGray(filename)
	cum_sum = integralBild(gray_im)

	newimage = np.empty((gray_im.shape[0],gray_im.shape[1]))
	for i in range(8,gray_im.shape[0] - 8):
		for j in range(8,gray_im.shape[1] - 8):
			#print gray_im[i-7:i+7,j-7:j+7]
			mean = (cum_sum[i-7][j-7] + cum_sum[i+7][j+7]  - cum_sum[i+7][j-7] - cum_sum[i-7][j+7])/(225)
			std = np.std(gray_im[i-7:i+7,j-7:j+7])
			treshhold = mean *(1+0.2*(std/128 - 1))
			#print "mean: ", mean, " std: ",std
			if gray_im[i][j] > treshhold:
				newimage[i][j] = 255
			else :
				newimage[i][j] = 0
	cv2.imwrite(filename + '_sauv2.jpg',newimage)
#niblack("34_res")

def rotate(filename):
	delta_y = 656 - 619.2
	delta_x = 212.2 - 440.6
	angle =  m.degrees(m.atan(delta_y / delta_x))

	img = cv2.imread(filename + '.jpg',0)
	rows,cols = img.shape
	img/2
	M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
	dst = cv2.warpAffine(img,M,(cols,rows))
	cv2.imwrite(filename + "_rot.jpg",dst)
def translate(filename):

	img = cv2.imread(filename + '.jpg',0)
	rows,cols = img.shape
	print rows,cols
	newimage = np.empty((rows,cols))
	for i in xrange(rows):
		for j in xrange(cols):
			newimage[i][(j+150)%cols] = img[i][j]
	cv2.imwrite(filename + "_transl.jpg",newimage)
translate("02_nib2")
#rotate("02_nib2")
#niblack2("25")

