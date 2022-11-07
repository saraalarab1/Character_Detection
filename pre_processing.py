import json
import os
import cv2 as cv
import json
from csv import reader
from features import horizontal_histogram_projection, horizontal_line_intersection, horizontal_ratio, horizontal_symmetry, nb_of_pixels_per_segment, vertical_histogram_projection, vertical_line_intersection, vertical_ratio, vertical_symmetry
WIDTH = 120
HEIGHT = 120
import numpy as np

def convert_csv_to_json():
    training_dataset = dict()
    # open file in read mode
    with open('english.csv', 'r') as read_obj:
        # pass the file object to reader() to get the reader object
        csv_reader = reader(read_obj)
        # Iterate over each row in the csv using reader object
        for row in csv_reader:
            # row variable is a list that represents a row in csv
            if os.path.exists('processed_images/' + row[0]):
                training_dataset[row[0]] = {"label": row[1]}
    with open('data.json', 'w', encoding='utf-8') as f:
        json.dump(training_dataset, f, ensure_ascii=False, indent=4)

def pre_process_images():
    images_dir = os.listdir('Img')
    with open('data.json', 'r') as f:
        data = json.load(f)
        for i in range(len(images_dir)):
            image_name = images_dir[i]
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
            print(image_name)
            image = cv.resize(image, dim, interpolation = cv.INTER_AREA)
            image = gray_to_black(image)
            cv.imwrite('processed_images_2/'+image_name, image)
    #         if image_name in data:
    #             data[image_name]['aspect_ratio'] = image.shape[1]/image.shape[0]
    # print('dumping data')
    # with open('data.json', 'w') as output:
    #        json.dump(data, output, ensure_ascii=False, indent = 4)

def extract_features_for_training_data():
    data = dict()
    with open('data.json', 'r') as f:
        data = json.load(f)
        for i in data.keys():
            if not os.path.exists('processed_images_2/' + i) or i not in data:
                continue
            print('processed_images/' + i)

            image= cv.imread('processed_images_2/' + i)
            # aspect_ratio_image = cv.imread('Img_2/' + i)
            # data[i]['aspect_ratio'] = aspect_ratio(aspect_ratio_image)

            data[i]['horizontal_histogram_projection'] = horizontal_histogram_projection(image)
            data[i]['horizontal_line_intersection'] = horizontal_line_intersection(image)
            data[i]['horizontal_ratio'] = horizontal_ratio(image)
            data[i]['horizontal_symmetry'] = horizontal_symmetry(image)
            data[i]['nb_of_pixels_per_segment'] = nb_of_pixels_per_segment(image, 7)
            data[i]['vertical_histogram_projection'] = vertical_histogram_projection(image)
            data[i]['vertical_line_intersection'] = vertical_line_intersection(image)
            data[i]['vertical_ratio'] = vertical_ratio(image)
            data[i]['vertical_symmetry'] = vertical_symmetry(image)
    print('dumping data')
    with open('data.json', 'w') as output:
           json.dump(data, output, ensure_ascii=False, indent = 4)


def get_total_nb_of_pixels():
        with open('data.json', 'r') as f:
            data = json.load(f)
            for i in data.keys():
                print('we are in: '+ i)
                current_image = cv.imread('skeletonized_images_2/' + i, 0)
                data[i]['nb_of_black_pixels'] =int(np.sum(current_image == 0))
            with open('data.json', 'w') as output:
                json.dump(data, output, ensure_ascii=False, indent = 4)

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

def gray_to_black(image):
    not_white_pixels = np.where(
    (image[:, :, 0] != 255) & 
    (image[:, :, 1] != 255) & 
    (image[:, :, 2] != 255)
    )
    image[not_white_pixels] = [0, 0, 0]
    return image


extract_features_for_training_data()
# get_total_nb_of_pixels()
# pre_process_images()
# convert_csv_to_json()
# post_skeletonization()
# create_json()
# assign_random_colors()
# plot()


