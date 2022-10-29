import json
import os
import random
import cv2 as cv
import numpy as np
from numpy import array, asarray
import matplotlib.pyplot as plt
from resizing import gray_to_black
from resizing import resize
import statistics
from mpl_toolkits.mplot3d import Axes3D
import json
from csv import reader
from features import get_aspect_ratio, get_horizontal_histogram_projection, get_horizontal_line_intersection, get_horizontal_ratio, get_horizontal_symmetry, get_nb_of_pixels_per_segment, get_vertical_histogram_projection, get_vertical_line_intersection, get_vertical_ratio, get_vertical_symmetry
from skeletonize import skeletonize_image
WIDTH = 40
HEIGHT = 40


def convert_csv_to_json():
    training_dataset = dict()
    # open file in read mode
    with open('english.csv', 'r') as read_obj:
        # pass the file object to reader() to get the reader object
        csv_reader = reader(read_obj)
        # Iterate over each row in the csv using reader object
        for row in csv_reader:
            # row variable is a list that represents a row in csv
            training_dataset[row[0]] = {"label": row[1]}
    with open('data.json', 'w', encoding='utf-8') as f:
        json.dump(training_dataset, f, ensure_ascii=False, indent=4)

def pre_process_images():
    images_dir = os.listdir('Img')
    for i in range(len(images_dir)):
        image_name = images_dir[i]
        print(image_name)
        image_path = os.path.join('Img', image_name)
        image = cv.imread(image_path)
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        thresh_image = cv.threshold(gray_image, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
        contours = cv.findContours(thresh_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        max_area = 0
        current_variables =  (0,0,0,0)
        dim = (WIDTH, HEIGHT)
        # choose bounding rectangle for character with biggest area
        for countour in contours:
            x,y,w,h = cv.boundingRect(countour)
            if w*h > max_area:
                max_area = w*h
                current_variables = (x,y,x+w,y+h)
        if current_variables != (0,0,0,0):
            # change image dimensions to minimum bounding rectangle
            image = image[current_variables[1]:current_variables[3], current_variables[0]:current_variables[2]] 
        # # resize image
        image = cv.resize(image, dim, interpolation = cv.INTER_AREA)
        image = gray_to_black(image)
        cv.imwrite('processed_images/'+image_name, image)



def extract_features_for_training_data():
    data = dict()
    with open('data.json', 'r') as f:
        data = json.load(f)
        for i in data.keys():
            print('processed_images/' + i)
            image= cv.imread('processed_images/' + i)
            # aspect_ratio_image = cv.imread('Img_2/' + i)
            # data[i]['aspect_ratio'] = get_aspect_ratio(aspect_ratio_image)

            data[i]['horizontal_histogram_projection'] = get_horizontal_histogram_projection(image)
            data[i]['horizontal_line_intersection'] = get_horizontal_line_intersection(image)
            data[i]['horizontal_ratio'] = get_horizontal_ratio(image)
            data[i]['horizontal_symmetry'] = get_horizontal_symmetry(image)

            data[i]['nb_of_pixels_per_segment'] = get_nb_of_pixels_per_segment(image, 7)

            data[i]['vertical_histogram_projection'] = get_vertical_histogram_projection(image)
            data[i]['vertical_line_intersection'] = get_vertical_line_intersection(image)
            data[i]['vertical_ratio'] = get_vertical_ratio(image)
            data[i]['vertical_symmetry'] = get_vertical_symmetry(image)
    with open('data.json', 'w') as output:
           json.dump(data, output, ensure_ascii=False, indent = 4)

# pre_process_images()
# convert_csv_to_json()

# extract_features_for_training_data()
# post_skeletonization()

# create_json()
# assign_random_colors()
# plot()
