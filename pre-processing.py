import json
import os
import random
import cv2 as cv
import numpy as np
from numpy import array, asarray
import matplotlib.pyplot as plt
WIDTH = 32
HEIGHT = 32
training_dataset = dict()

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
        image_path = os.path.join('Img', image_name)
        img = cv.imread(image_path)
        gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
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
            img = img[current_variables[1]:current_variables[3], current_variables[0]:current_variables[2]] 
            #get_aspect_ratio
            get_aspect_ratio(img, image_name)
            # resize image
            img = cv.resize(img, dim, interpolation = cv.INTER_AREA)
            # Feature 1
            get_vertical_symmetry_feature(img, image_name)
            # Feature 2
            get_horizontal_symmetry_feature(img, image_name)
            # Feature 3
            get_vertical_percentage_feature(img,image_name)
            # Feature 4
            get_horizontal_percentage_feature(img, image_name)


def get_vertical_symmetry_feature(img, image_name):
    image_left = img[:,:int(WIDTH/2)]
    image_right = cv.flip(img[:, int(WIDTH/2):],1)
    # cv.imshow('Left', image_left)
    # cv.imshow('Right', image_right)
    # cv.waitKey(0)
    diff = cv.subtract(image_left, image_right)
    err = np.sum(diff**2)
    mse = err/(float(HEIGHT*WIDTH))
    image_info = training_dataset[image_name]
    image_info["feature_vertical_symmetry"] = mse
    training_dataset[image_name] = image_info

def get_horizontal_symmetry_feature(img, image_name):
    image_top = img[: int(WIDTH/2), :]
    image_bottom =  cv.flip(img[ int(WIDTH/2):, :],0)
    #cv.imshow('Top', image_top)
    #cv.imshow('Bottom', image_bottom)
    #cv.waitKey(0)
    diff = cv.subtract(image_top, image_bottom)
    err = np.sum(diff**2)
    mse = err/(float(HEIGHT*WIDTH))
    image_info = training_dataset[image_name]
    image_info["feature_horizontal_symmetry"] = mse
    training_dataset[image_name] = image_info

def get_vertical_percentage_feature(img, image_name):
    image_left = img[:,:int(WIDTH/2)]
    image_right = cv.flip(img[:, int(WIDTH/2):],1)
    # cv.imshow('Left', image_left)
    # cv.imshow('Right', image_right)
    # cv.waitKey(0)
    total_pixel_left = np.count_nonzero(np.all(image_left == [0,0,0], axis = 2))
    total_pixel_right = np.count_nonzero(np.all(image_right == [0,0,0], axis = 2))
    ratio_percentage = total_pixel_left/total_pixel_right
    image_info = training_dataset[image_name]
    image_info["feature_vertical_ratio"] = ratio_percentage
    training_dataset[image_name] = image_info

def get_horizontal_percentage_feature(img, image_name):
    image_top = img[: int(WIDTH/2), :]
    image_bottom =  cv.flip(img[ int(WIDTH/2):, :],0)
    #cv.imshow('Top', image_top)
    #cv.imshow('Bottom', image_bottom)
    #cv.waitKey(0)
    total_pixel_top = np.count_nonzero(np.all(image_top == [0,0,0], axis = 2))
    total_pixel_bottom = np.count_nonzero(np.all(image_bottom == [0,0,0], axis = 2))
    ratio_percentage = total_pixel_top/total_pixel_bottom
    image_info = training_dataset[image_name]
    image_info["feature_horizontal_ratio"] = ratio_percentage
    training_dataset[image_name] = image_info


def get_aspect_ratio(image, image_name):
    image_info = training_dataset[image_name]
    image_info["aspect_ratio"] =  float(image.shape[1]) / image.shape[0]
    training_dataset[image_name] = image_info

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
            zdata.append(data[i]["feature_horizontal_ratio"])
            ydata.append(data[i]["feature_vertical_ratio"])
            xdata.append(data[i]["aspect_ratio"])
            colors.append(data[i]['color'])
            # plt.scatter(data[i]["feature_horizontal_ratio"], data[i]["feature_vertical_ratio"], c= data[i]["color"], s= 5)
        # plt.show()
        print(colors)
        ax.scatter3D(xdata, ydata, zdata, c=colors)
        plt.show()

read_csv()
pre_process_images()
create_json()
assign_random_colors()
plot()
