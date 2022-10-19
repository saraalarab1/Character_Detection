import os
import cv2 as cv
import numpy as np

from skeletonize import skeletonize_image 

def resize(image, image_name):
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    thresh_image = cv.threshold(gray_image, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
    contours = cv.findContours(thresh_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    max_area = 0
    current_variables =  (0,0,0,0)
    dim = (40, 40)
    # choose bounding rectangle for character with biggest area
    for countour in contours:
        x,y,w,h = cv.boundingRect(countour)
        if w*h > max_area:
            max_area = w*h
            current_variables = (x,y,x+w,y+h)
    if current_variables != (0,0,0,0):
        # change image dimensions to minimum bounding rectangle
        image = image[current_variables[1]:current_variables[3], current_variables[0]:current_variables[2]] 
    print(image.shape)
    image = cv.resize(image, (40, 40), interpolation = cv.INTER_AREA)
    cv.imwrite('skeletonized_new/'+image_name, image)
# cv.imshow('image', image)
# cv.waitKey(0)


def gray_to_black(image):
    not_white_pixels = np.where(
    (image[:, :, 0] != 255) & 
    (image[:, :, 1] != 255) & 
    (image[:, :, 2] != 255)
    )
    image[not_white_pixels] = [0, 0, 0]
    return image


def post_skeletonization():
    images_path = 'skeletonized_images'
    images = os.listdir(images_path)
    for image in images:
        image_path = os.path.join(images_path, image)
        current_image = cv.imread(image_path)
        # print(current_image)
        
        green_pixels = np.where(
        (current_image[:, :, 0] == 0) & 
        (current_image[:, :, 1] == 255) & 
        (current_image[:, :, 2] == 0)
        )
        black_pixels = np.where(
        (current_image[:, :, 0] == 0) & 
        (current_image[:, :, 1] == 0) & 
        (current_image[:, :, 2] == 0)
        )
        current_image[green_pixels] = [0, 0, 0]
        current_image[black_pixels] = [255, 255, 255]
        gray_image = cv.cvtColor(current_image, cv.COLOR_BGR2GRAY)
        thresh_image = cv.threshold(gray_image, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
        contours = cv.findContours(thresh_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]

        max_area = 0
        current_variables =  (0,0,0,0)
        dim = (40, 40)
        # choose bounding rectangle for character with biggest area
        for countour in contours:
            x,y,w,h = cv.boundingRect(countour)
            if w*h > max_area:
                max_area = w*h
                current_variables = (x,y,x+w,y+h)
        if current_variables != (0,0,0,0):
            # change image dimensions to minimum bounding rectangle
            current_image = current_image[current_variables[1]:current_variables[3], current_variables[0]:current_variables[2]] 
        current_image = cv.resize(current_image, dim, interpolation = cv.INTER_AREA)
        not_white_pixels = np.where(
        (current_image[:, :, 0] != 255) & 
        (current_image[:, :, 1] != 255) & 
        (current_image[:, :, 2] != 255)
        )
        current_image[not_white_pixels] = [0, 0, 0]
        cv.imwrite('skeletonized_cropped/'+image, current_image)