import json
import os
import random
import cv2 as cv
import numpy as np
from numpy import array, asarray
import matplotlib.pyplot as plt

from skeletonize import skeletonize_image
WIDTH = 32
HEIGHT = 32
training_dataset = dict()
MAX_COUNT_HORIZONTAL = 5
MAX_COUNT_VERTICAL = 4

def read_csv():
    from csv import reader
    # open file in read mode
    with open('english.csv', 'r') as read_obj:
        # pass the file object to reader() to get the reader object
        csv_reader = reader(read_obj)
        # Iterate over each row in the csv using reader object
        for row in csv_reader:
            # row variable is a list that represents a row in csv
            training_dataset[row[0]] = {"label": row[1]}

def pre_process_images():
    images_dir = os.listdir('Img')
    for i in range(len(images_dir)):
        image_name = images_dir[i]
        image_path = os.path.join('Img_2', image_name)
        image = cv.imread(image_path)
        skeletonize_image(image, image_name)
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
            # Feature 1
            training_dataset[image_name]["aspect_ratio"] = get_aspect_ratio(image)
            # resize image
            image = cv.resize(image, dim, interpolation = cv.INTER_AREA)
            # Feature 2
            training_dataset[image_name]["vertical_ratio"] = get_vertical_ratio(image)
            # Feature 3
            training_dataset[image_name]["horizontal_ratio"] = get_horizontal_ratio(image)
            # Feature 4
            training_dataset[image_name]["vertical_symmetry"] = get_vertical_symmetry(image)
            # Feature 5
            training_dataset[image_name]["horizontal_symmetry"] = get_horizontal_symmetry(image)
            # Feature 6
            training_dataset[image_name]["percentage_of_pixels_at_horizontal_center"] = percentage_of_pixels_on_horizontal_center(image)
            # Feature 7 
            training_dataset[image_name]["percentage_of_pixels_at_vertical_center"] = percentage_of_pixels_on_vertical_center(image)
            # Feature 8
            training_dataset[image_name]["horizontal_line_intersection_count"] = get_horizontal_line_intersection(image)
            # Feature 9
            training_dataset[image_name]["vertical_line_intersection_count"] = get_vertical_line_intersection(image)
            # Feature 10
            training_dataset[image_name]["vertical_histogram_projection"] = get_vertical_histogram_projection(image)
            # Feature 11
            training_dataset[image_name]["horizontal_histogram_projection"] = get_horizontal_histogram_projection(image)

def get_aspect_ratio(image):
    """
    Add definition
    """
    return  float(image.shape[1]) / image.shape[0]

def get_vertical_ratio(image):
    """
    Add definition
    """
    image_left = image[:,:int(WIDTH/2)]
    image_right = image[:, int(WIDTH/2):]
    total_pixel_left = np.sum(image_left == 0)
    total_pixel_right =  np.sum(image_right == 0)
    return total_pixel_left/total_pixel_right

def get_horizontal_ratio(image):
    """
    Add definition
    """
    image_top = image[:int(HEIGHT/2), :]
    image_bottom =  image[int(HEIGHT/2):, :]
    total_pixel_top = np.sum(image_top == 0)
    total_pixel_bottom = np.sum(image_bottom == 0)
    return total_pixel_top/total_pixel_bottom

def get_vertical_symmetry(image):
    """
    Add definition
    """
    image_left = image[:,:int(WIDTH/2)]
    image_right = cv.flip(image[:, int(WIDTH/2):],1)
    diff = cv.subtract(image_left, image_right)
    white_pixels = np.sum(diff == 255)
    black_pixels = np.sum(diff == 0)
    return white_pixels/(black_pixels+white_pixels)

def get_horizontal_symmetry(image):
    """
    Add definition
    """
    image_top = image[: int(HEIGHT/2), :]
    image_bottom =  cv.flip(image[ int(HEIGHT/2):, :],0)
    diff = cv.subtract(image_top, image_bottom)
    white_pixels = np.sum(diff == 255)
    black_pixels = np.sum(diff == 0)
    return white_pixels/(black_pixels+white_pixels)

def get_horizontal_line_intersection(image):
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

def get_vertical_line_intersection(image):
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

def get_vertical_histogram_projection(image):
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    thresh_image = cv.threshold(gray_image, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
    vertical_pixel_sum = np.sum(thresh_image, axis=0)
    return vertical_pixel_sum.tolist()

def get_horizontal_histogram_projection(image):
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    thresh_image = cv.threshold(gray_image, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
    horizontal_pixel_sum = np.sum(thresh_image, axis=1)
    return horizontal_pixel_sum.tolist()


def create_json():
    import json
    with open('data.json', 'w', encoding='utf-8') as f:
        json.dump(training_dataset, f, ensure_ascii=False, indent=4)

def assign_random_colors():
    label_color = dict()
    with open('data.json', 'r') as f:
        data = json.load(f)
        for i in data.keys():
            if not data[i]["label"] in label_color.keys():
                label_color[data[i]["label"]] = "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
            data[i]["color"] = label_color[data[i]["label"]]
        with open('data_with_colors.json', 'w') as output:
            json.dump(data, output, ensure_ascii=False, indent = 4)


def percentage_of_pixels_on_horizontal_center(image):
    """
    This function returns the percentage of non-white pixels 
    along the horizontal axis at the center of an image
    """
    total_nb_of_black_pixels = len(np.where((image[:, :, 0]==0) & (image[:, :, 1]==0) & (image[:, :, 2]==0))[0])
    nb_of_black_pixels_at_horizontal = len([i for i in image[int(WIDTH/2)] if tuple(i) == (0, 0, 0)])
    return nb_of_black_pixels_at_horizontal/total_nb_of_black_pixels 

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
    return nb_of_pixels_at_vertical/total_nb_of_black_pixels

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
        # cv.imshow('image', current_image)
        # cv.waitKey(0)
        cv.imwrite('skeletonized_cropped/'+image, current_image)
        # break
# post_skeletonization()
def plot():
    with open('data_with_colors.json', 'r') as f:
        data = json.load(f)
        index = 0
        fig = plt.figure()
        ax = plt.axes(projection = '3d')
        zdata = []
        ydata = []
        xdata = []
        colors = []
        for i in data.keys():
            index = index + 1
            if index == 1000:
                break
            zdata.append(data[i]["horizontal_symmetry"])
            ydata.append(data[i]["vertical_ratio"])
            xdata.append(data[i]["aspect_ratio"])
            colors.append(data[i]['color'])
        ax.scatter3D(xdata, ydata, zdata, c=colors)
        plt.show()

read_csv()
pre_process_images()
create_json()
assign_random_colors()
plot()
