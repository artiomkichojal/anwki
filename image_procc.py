#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import numpy as np
import cv2
import math as m
import matplotlib.pyplot as plt


def imgToGray(filename):
	image = cv2.imread(filename+'.jpg')
	# print "processing picture to gray"
	# cv2.imwrite(filename + '_g.jpg',np.mean(image,axis=2))
	# print "to Gray ready"
	return np.mean(image,axis=2)

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
create image with width
"""
def resizeIm(filename,width):
	oriimage = cv2.imread(filename + '.jpg')
	if width < oriimage.shape[1]:
		faktor = float(float(width)/float(oriimage.shape[1]))
		newimage = cv2.resize(oriimage,(0,0),fx=faktor, fy=faktor)
		cv2.imwrite(filename + '_res.jpg',newimage)
		return True
	return False
"""
gibt integralmatrix zurueck
"""
def integralBild(data_image):
	return np.cumsum(np.cumsum(data_image,axis=0),axis=1)

def niblack2(filename):
	#verkleinere bild bis breite = 600, falls bildbreite groesser als 600
	#und erstelle graues bild
	if resizeIm(filename, 600):
		gray_im = imgToGray(filename+"_res")
	else:
		gray_im = imgToGray(filename)
	cum_sum = integralBild(gray_im)
	#initialise result matrix
	newimage = np.empty((gray_im.shape[0],gray_im.shape[1]))
	for i in range(8,gray_im.shape[0] - 7):
		for j in range(8,gray_im.shape[1] - 7):
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
	#verkleinere bild bis breite = 600, falls bildbreite groesser als 600
	#und erstelle graues bild
	if resizeIm(filename, 600):
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

def rotate(filename):
	delta_y = 656 - 619.2
	delta_x = 212.2 - 440.6
	angle =  m.degrees(m.atan(delta_y / delta_x))

	img = cv2.imread(filename + '.jpg',0)
	rows,cols = img.shape
	M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
	dst = cv2.warpAffine(img,M,(cols,rows))
	cv2.imwrite(filename + "_rot.jpg",dst)

def rotateManuell(filename):
	delta_y = 656 - 619.2
	delta_x = 212.2 - 440.6
	angle =  m.atan(delta_y / delta_x)

	img = cv2.imread(filename + '.jpg',0)
	rows,cols = img.shape
	rot_matr = np.array([[np.cos(angle),-np.sin(angle),0],
						[np.sin(angle),np.cos(angle),0],
						[0,0,1]])
	#print np.dot(rot_matr,np.array([0,0,1]))
	newimage = np.empty((rows,cols))
	for i in xrange(rows):
		for j in xrange(cols):
			temp = np.dot(rot_matr,np.array([i,j,1]))
			if temp[0] >= 0 and temp[1] >= 0 :
				newimage[temp[0]%rows][temp[1]%cols] = img[i][j]			
	cv2.imwrite(filename + "_rotM1.jpg",newimage)
def translate(filename):

	img = cv2.imread(filename + '.jpg',0)
	rows,cols = img.shape
	
	newimage = np.empty((rows,cols))
	for i in xrange(rows):
		for j in xrange(cols):
			newimage[i][(j+150)%cols] = img[i][j]
	cv2.imwrite(filename + "_transl.jpg",newimage)
#translate("02_nib2")

"""
transformiert das bild per homographie
"""
def transformHomography(filename):
	img = cv2.imread(filename + '.png',0)
	rows,cols = img.shape
	print rows,cols
	pts1 = np.float32([[429,66],[540,1656],[1821,69],[2127,1413]])
	pts2 = np.float32([[0,0],[0,rows],[cols,0],[cols,rows]])
	#pts2 = np.float32([[0,0],[1590,0],[0,1400],[1590,1400]])

	M = cv2.getPerspectiveTransform(pts1,pts2)

	dst = cv2.warpPerspective(img,M,(cols,rows))
	plt.subplot(122),plt.imshow(dst),plt.title('Output')
	#plt.show()
	cv2.imwrite(filename + "_norm.png",dst)
transformHomography("02")
#rotate("02_nib2")
#sauvola2("02")
#rotateManuell("02_sauv2")
"""
pts_vector1 =  4 ecken start
pts_vector2 = 4 ecken finish
"""
def solveDLT(pts_vector1,pts_vector2):
	# h = np.array([[-429,-66,-1,0,0,0,0,0,0],
	# 				[-540,-1656,-1,0,0,0,0,0,0],
	# 				[-1821,-69,-1,0,0,0,1821*2592,2592*69,2592],
	# 				[-2127,1413,-1,0,0,0,1728*2127,2127*25921,1728],
	# 				[0,0,0,0,0,-1,0,0,0],
	# 				[0,0,0,0,-1728,-1,1728*540,1728*1656,1728],
	# 				[0,0,0,-2592,0,-1,0,0,0],
	# 				[0,0,0,-1728,-2592,-1,2592*2127,2592*1413,2592]])
	# b = np.array([0,0,0,0,0,0,0,0])
	# print b.shape
	# print np.linalg.lstsq(h,b)[0]

	#8x9 matrix (9 unbekannten <- h11 ... h33)
	x_0 = pts_vector1[0][0]
	y_0 = pts_vector1[0][1]
	x_new_0 = pts_vector2[0][0]
	y_new_0 = pts_vector2[0][1]
	a = np.array([[-x_0,-y_0,-1,0,0,0,x_new_0*x_0,x_new_0*y_0,x_new_0],
					[0,0,0,-x_0,-y_0,-1,y_new_0*x_0,y_new_0*y_0,y_new_0]])
	#a = np.empty((2,))
	for i in range(1,3):
		print i
		x_i = pts_vector1[i][0]
		y_i = pts_vector1[i][1]
		x_new_i = pts_vector2[i][0]
		y_new_i = pts_vector2[i][1]
		x_row = np.array([-x_0,-y_0,-1,0,0,0,x_new_i*x_i,x_new_i*y_i,x_new_i])
		y_row = np.array([-x_0,-y_0,-1,0,0,0,x_new_i*x_i,x_new_i*y_i,x_new_i])
		a = np.vstack([a,x_row])

		a = np.vstack([a,y_row])
		#print a
	print a
	b = np.array([0,0,0,0,0,0,0,0])
	return np.linalg.lstsq(a,b)[0]
solveDLT(np.float32([[429,66],[540,1656],[1821,69],[2127,1413]]),
	np.float32([[0,0],[0,1728],[2592,0],[2592,1728]]))

a = np.array([[3,1], [1,2]])
b = np.array([0,0])
#print np.linalg.lstsq(a, b)

