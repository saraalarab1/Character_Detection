import json
import os
import random
import cv2 as cv
import numpy as np
from numpy import array, asarray

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


def get_error():
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
            # resize image
            img = cv.resize(img, dim, interpolation = cv.INTER_AREA)
            image_left = img[:,:int(WIDTH/2)]
            image_right = cv.flip(img[:, int(WIDTH/2):],1)
            diff = cv.subtract(image_left, image_right)
            err = np.sum(diff**2)
            mse = err/(float(HEIGHT*WIDTH))
            img= image_left + image_right
            image_info = training_dataset[image_name]
            image_info["feature_vertical_split"] = mse
            training_dataset[image_name] = image_info
        

def create_json():
    import json
    with open('data.json', 'w', encoding='utf-8') as f:
        json.dump(training_dataset, f, ensure_ascii=False, indent=4)
# read_csv()
# get_error()
# create_json()
# print (training_dataset)

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

# assign_random_colors()

def draw():
    with open('data_with_colors.json', 'r') as f:
        data = json.load(f)
        import matplotlib.pyplot as plt
        index = 0
        for i in data.keys():
            index = index + 1
            if index == 100:
                break
            plt.scatter(data[i]["feature_vertical_split"], random.randint(0,50), c= data[i]["color"], s= 5)
        plt.show()
draw()