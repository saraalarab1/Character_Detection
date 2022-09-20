import os 
import cv2 as cv
import numpy as np
from numpy import asarray
images = os.listdir('Img')

for image_name in images: 
    image_path = os.path.join('Img', image_name)
    img = cv.imread(image_path)
    gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    thresh_image = cv.threshold(gray_image, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
    contours = cv.findContours(thresh_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    max_area = 0
    current_variables =  (0,0,0,0)
    for i in contours:
        x,y,w,h = cv.boundingRect(i)
        if w*h > max_area:
            max_area = w*h
            current_variables = (x,y,x+w,y+h)
    if current_variables != (0,0,0,0):
        img = img[current_variables[1]:current_variables[3], current_variables[0]:current_variables[2]]
    cv.imwrite('Img_2/' + image_name, img)
    
