import cv2
import numpy as np

green=(0,255,0)
red=(255,0,0)
blue=(0,0,255)

def findContour(image): #cari garis tepi terbesar
	image=image.copy()

	_ , contours , hierarchy=cv2.findContours(image,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
	contour_sizes=[(cv2.contourArea(contour),contour) for contour in contours]
	biggest_contour=max(contour_sizes,key=lambda x:x[0])[1]
	mask=np.zeros(image.shape,np.uint8)
	cv2.drawContours(mask,[biggest_contour],-1,255,-1)
	cv2.imshow('mask',mask)
	return biggest_contour,mask

def circleContour(image,contour): #make apple circled
	image_with_ellipse=image.copy()
	ellipse=cv2.fitEllipse(contour)
	cv2.ellipse(image_with_ellipse,ellipse,green,2,1)
	imageEllipse = cv2.cvtColor(image_with_ellipse,cv2.COLOR_RGB2BGR)
	return imageEllipse

def drawApple(image):
	#PRE PROCESSING OF IMAGE
	image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
	maxsize=max(image.shape)
	
	#resize image
	scale=700/maxsize
	image=cv2.resize(image,None,fx=scale,fy=scale)
	image_blur=cv2.GaussianBlur(image,(7,7),0) #blurring
	image_blur_hsv=cv2.cvtColor(image_blur,cv2.COLOR_RGB2HSV) #HSV for color
	
        #HSV for apple detection
	min_color=np.array([0,100,100])
	max_color=np.array([10,256,256])
	mask1=cv2.inRange(image_blur_hsv,min_color,max_color)
	min_color2=np.array([170,100,80])
	max_color2=np.array([180,256,256])
	mask2=cv2.inRange(image_blur_hsv,min_color2,max_color2)
	mask=mask1+mask2
	
	big_contour,mask_fruit=findContour(mask) #get contour
	circled=circleContour(image,big_contour) #make image circled
	return circled

#input image
apple=cv2.imread('apel2.jpg')

#process image
result_apple=drawApple(apple)

#output image
cv2.imshow('Original', apple)
cv2.imshow('hasil',result_apple)

#cv2.imwrite('red_new.jpg',result_apple)
