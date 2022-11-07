
import json
import cv2 as cv
import numpy as np
from pre_processing import gray_to_black
MAX_COUNT_HORIZONTAL = 5
MAX_COUNT_VERTICAL = 4
WIDTH = 120
HEIGHT = 120

def nb_of_pixels_per_segment(image, index):
    """
    Add definition
    """
    pixels_per_segment = []
    for i in range(0, WIDTH, index):
        for j in range(0, HEIGHT, index):
            total_pixels = 0
            for k in range(0, min(WIDTH-i,index)):
                for l in range(0, min(HEIGHT-j,index)):
                    if tuple(image[i+k][j+l])!=(255, 255, 255):
                        total_pixels = total_pixels + 1
            pixels_per_segment.append(total_pixels)
    return pixels_per_segment

def aspect_ratio(image):
    """
    Add definition
    """
    return  round(float(image.shape[1]) / image.shape[0],4)

def vertical_ratio(image):
    """
    Add definition
    """
    image_left = image[:,:int(WIDTH/2)]
    image_right = image[:, int(WIDTH/2):]
    total_pixel_left = np.sum(image_left == 0)
    total_pixel_right =  np.sum(image_right == 0)
    return round(total_pixel_left/total_pixel_right,4)

def horizontal_ratio(image):
    """
    Add definition
    """
    image_top = image[:int(HEIGHT/2), :]
    image_bottom =  image[int(HEIGHT/2):, :]
    total_pixel_top = np.sum(image_top == 0)
    total_pixel_bottom = np.sum(image_bottom == 0)
    return round(total_pixel_top/total_pixel_bottom,4)

def vertical_symmetry(image):
    """
    Add definition
    """
    image_left = image[:,:int(WIDTH/2)]
    image_right = cv.flip(image[:, int(WIDTH/2):],1)
    diff = cv.subtract(image_left, image_right)
    white_pixels = np.sum(diff == 255)
    black_pixels = np.sum(diff == 0)
    return round(white_pixels/(black_pixels+white_pixels),4)

def horizontal_symmetry(image):
    """
    Add definition
    """
    image_top = image[: int(HEIGHT/2), :]
    image_bottom =  cv.flip(image[ int(HEIGHT/2):, :],0)
    diff = cv.subtract(image_top, image_bottom)
    white_pixels = np.sum(diff == 255)
    black_pixels = np.sum(diff == 0)
    return round(white_pixels/(black_pixels+white_pixels),4)

def horizontal_line_intersection(image):
    line = int(image.shape[0]/3)
    w = image.shape[1]
    intersection_count =0
    x=0

    # loop over the horizontal line, pixel by pixel
    while(x<w-1):
        x+=1
        value = image[line][x]
        # if value different than white pixel 
        if (value != 255).all():
            while (value != 255).all() and x<w-1:
                x+=1
                value = image[line][x]
            intersection_count+=1

    return intersection_count/MAX_COUNT_HORIZONTAL

def vertical_line_intersection(image):
    line = int(image.shape[1]/3)
    h = image.shape[0]
    intersection_count =0
    y=0

    # loop over the horizontal line, pixel by pixel
    while(y<h-1):
        y+=1
        value = image[y][line]
        # if value different than white pixel 
        if (value != 255).all():
            while (value != 255).all() and y<h-1:
                y+=1
                value = image[y][line]
            intersection_count+=1
    return intersection_count/MAX_COUNT_VERTICAL

def vertical_histogram_projection(image):
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    thresh_image = cv.threshold(gray_image, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
    vertical_pixel_sum = np.sum(thresh_image, axis=0)
    smaller_vector=[]
    sum=0
    for i in range(len(vertical_pixel_sum)):
        sum+=vertical_pixel_sum[i]
        if i%4==0:
            smaller_vector.append(int(sum))
            sum=0
    return smaller_vector

def horizontal_histogram_projection(image):
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    thresh_image = cv.threshold(gray_image, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
    horizontal_pixel_sum = np.sum(thresh_image, axis=1)
    smaller_vector=[]
    sum=0
    for i in range(len(horizontal_pixel_sum)):
        sum+=horizontal_pixel_sum[i]
        if i%4==0:
            smaller_vector.append(int(sum))
            sum=0
    return smaller_vector

def percentage_of_pixels_on_horizontal_center(image):
    """
    This function returns the percentage of non-white pixels 
    along the horizontal axis at the center of an image
    """
    total_nb_of_black_pixels = len(np.where((image[:, :, 0]==0) & (image[:, :, 1]==0) & (image[:, :, 2]==0))[0])
    nb_of_black_pixels_at_horizontal = len([i for i in image[int(WIDTH/2)] if tuple(i) == (0, 0, 0)])
    return nb_of_black_pixels_at_horizontal/(total_nb_of_black_pixels if total_nb_of_black_pixels > 0 else 0.1)

def percentage_of_pixels_on_vertical_center(image):
    """
    This function returns the percentage of non-white pixels
    along the vertical axis at the center of an image"""
    nb_of_pixels_at_vertical = 0
    total_nb_of_black_pixels = len(np.where((image[:, :, 0]==0) & (image[:, :, 1]==0) & (image[:, :, 2]==0))[0])
    for i in range(WIDTH):
        if tuple(image[i][int(HEIGHT/2)]) == (0, 0, 0):
            nb_of_pixels_at_vertical = nb_of_pixels_at_vertical + 1
        image[i][int(HEIGHT/2)] = (255, 0, 0)
    return nb_of_pixels_at_vertical/(total_nb_of_black_pixels if total_nb_of_black_pixels > 0 else 0.1)

def get_character_features(features, characters):
    """
    This function gets required features of a certain character
    """
    features_data = []
    features_of_characters = []
    for character in characters: 
        for feature in features:
            if feature == 'nb_of_pixels_per_segment':
                features_data.append(nb_of_pixels_per_segment(character, 7))
            elif feature == 'aspect_ratio':
                features_data.append(aspect_ratio(character))
            elif feature == 'vertical_ratio':
                features_data.append(vertical_ratio(character))
            elif feature == 'horizontal_ratio':
                features_data.append(horizontal_ratio(character))
            elif feature == 'vertical_symmetry':
                features_data.append(vertical_symmetry(character))
            elif feature == 'horizontal_symmetry':
                features_data.append(horizontal_symmetry(character))
            elif feature == 'horizontal_line_intersection':
                features_data.append(horizontal_line_intersection(character))
            elif feature == 'vertical_line_intersection':
                features_data.append(vertical_line_intersection(character))
            elif feature == 'vertical_histogram_projection':
                features_data.append(vertical_histogram_projection(character))
            elif feature == 'horizontal_histogram_projection':
                features_data.append(horizontal_histogram_projection(character))
            elif feature == 'percentage_of_pixels_on_horizontal_center':
                features_data.append(percentage_of_pixels_on_horizontal_center(character))
            elif feature == 'percentage_of_pixels_on_vertical_center':
                features_data.append(percentage_of_pixels_on_vertical_center(character))
        features_of_characters.append(features_data)
        features_data = []
    return features_of_characters

def pre_process_image(image):
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    thresh_image = cv.threshold(gray_image, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
    contours = cv.findContours(thresh_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    max_area = 0
    current_variables =  (0,0,0,0)
    dim = (WIDTH, HEIGHT)
    # choose bounding rectangle for character with biggest area
    letters = []
    all_contours = []
    for countour in contours:
        x,y,w,h = cv.boundingRect(countour)
        # if w*h > max_area:
        #     max_area = w*h
        current_variables = (x,y,x+w,y+h)
        all_contours.append(current_variables)
        # if current_variables != (0,0,0,0):
        # # change image dimensions to minimum bounding rectangle
        #     current_image = image[current_variables[1]:current_variables[3], current_variables[0]:current_variables[2]]
        #     current_image = cv.resize(current_image, dim, interpolation = cv.INTER_AREA)
        #     current_image = gray_to_black(current_image)
        #     letters.append(current_image)
    new_countours = []
    for i in range(len(all_contours)):
        if i >= len(all_contours):
            break
        x1,y1, x1_, y1_ = all_contours[i]
        found = False
        print(i)
        for j in range(i+1, len(all_contours)):
            x2, y2, x2_, y2_ = all_contours[j]
            print(x1, x1_)
            print(x2, x2_)
            print('---------------------')
            if (x1 >x2 and x1 < x2_) or (x1_<x2_ and x1_ >x2) or (x2 > x1 and x2 < x1_) or (x2_ < x1_ and x2_ > x1):
                new_countours.append((min(x1,x2), min(y1, y2), max(x1_, x2_), max(y1_, y2_)))
                found = True
                print('found one')
                all_contours.pop(j)
                break
        if not found:
            new_countours.append(all_contours[i])
    # current_variables = new_countours[0]
    # cv.imshow('image', image[current_variables[1]:current_variables[3], current_variables[0]:current_variables[2]])
    # cv.waitKey(0)
    # # resize image
    new_countours.sort(key=lambda x: x[0], reverse=False)
    for contour in new_countours: 
        current_variables = contour
        current_image = image[current_variables[1]:current_variables[3], current_variables[0]:current_variables[2]]
        current_image = cv.resize(current_image, dim, interpolation = cv.INTER_AREA)
        current_image = gray_to_black(current_image)
        letters.append(current_image)

    return letters