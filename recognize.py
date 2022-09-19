import os 
import cv2 as cv
import numpy as np
from numpy import asarray
images = os.listdir('Img')
for image_name in images: 
    image_path = os.path.join('Img', image_name)
    current_image = cv.imread(image_path)
    image_array = cv.cvtColor(current_image, cv.COLOR_BGR2RGB)
    found = False
    for i in range(len(image_array)):
        for j in range(len(image_array[i])):
            if tuple(image_array[i][j]) != (255, 255, 255): # black pixel found 
                for k in range(len(image_array[i])):
                    image_array[i][k] = (0, 255, 0)
                found = True
                break
        if found: 
            break
    found = False
    for i in range(len(image_array)-1, 0, -1):
        for j in range(len(image_array[i])):
            if tuple(image_array[i][j]) != (255, 255, 255):
                for k in range(len(image_array[i])):
                    image_array[i][k] = (0, 255, 0)
                found = True
                break
        if found: 
            break          
    cv.imwrite('new_image.png', image_array)
    break