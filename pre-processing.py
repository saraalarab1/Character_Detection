import os 
import cv2 as cv
import numpy as np
from numpy import array, asarray
images_dir = os.listdir('Img')

WIDTH = 32
HEIGHT = 32

for image_name in images_dir:
    image_name = 'img030-003.png'
    image_path = os.path.join('Img', image_name)
    img = cv.imread(image_path)
    array_created = np.full((50, 50, 3),
                        255, dtype = np.uint8)
    cv.rectangle(array_created, (10,10), (40,40), (255, 0, 255), 1)   
    cv.imwrite('Img_2/'+ image_name, array_created)
    # img = array_created
    gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    thresh_image = cv.threshold(gray_image, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
    contours = cv.findContours(thresh_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    max_area = 0
    current_variables =  (0,0,0,0)
    dim = (WIDTH, HEIGHT)
    # choose bounding rectangle for character with biggest area
    for i in contours:
        x,y,w,h = cv.boundingRect(i)
        if w*h > max_area:
            max_area = w*h
            current_variables = (x,y,x+w,y+h)
    if current_variables != (0,0,0,0):
        # change image dimensions to minimum bounding rectangle
        img = img[current_variables[1]:current_variables[3], current_variables[0]:current_variables[2]] 
        # resize image
        img = cv.resize(img, dim, interpolation = cv.INTER_AREA)
        image_left = img[:,:int(WIDTH/2)]
        image_right = cv.flip(img[:, int(WIDTH/2):],1)
        diff = cv.subtract(image_left, image_right)
        err = np.sum(diff**2)
        print(err)
        print(diff.flatten()[0:100])
        mse = err/(float(HEIGHT*WIDTH))
        img= image_left + image_right
        cv.imwrite('Img_3/left_' + image_name, image_left)
        cv.imwrite('Img_3/right_' + image_name, image_right)

        # img = cv.flip(img[:, int(WIDTH/2):],1)
        print(mse)
        cv.imwrite('Img_3/' + image_name, img)
    break
    
