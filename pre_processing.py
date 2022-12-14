import json
import os
import cv2 as cv
from resizing import gray_to_black
import json
from csv import reader
from features import horizontal_histogram_projection, horizontal_line_intersection, horizontal_ratio, horizontal_symmetry, nb_of_pixels_per_segment, vertical_histogram_projection, vertical_line_intersection, vertical_ratio, vertical_symmetry
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
            if os.path.exists('new_Img/' + row[0]):
                training_dataset[row[0]] = {"label": row[1]}
    with open('data.json', 'w', encoding='utf-8') as f:
        json.dump(training_dataset, f, ensure_ascii=False, indent=4)

def pre_process_images():
    images_dir = os.listdir('new_Img')
    with open('data.json', 'r') as f:
        data = json.load(f)
        for i in range(len(images_dir)):
            image_name = images_dir[i]
            image_path = os.path.join('new_Img', image_name)
            image = cv.imread(image_path)
            gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            thresh_image = cv.threshold(gray_image, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
            contours = cv.findContours(thresh_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            contours = contours[0] if len(contours) == 2 else contours[1]
            max_area = 0
            current_variables =  (0,0,0,0)
            dim = (WIDTH, HEIGHT)
            # choose bounding rectangle for character with biggest area
            if data[image_name]['label'] == 'i' or data[image_name]['label'] == 'j':
                if len(contours) == 2:
                    x1,y1,w1,h1 = cv.boundingRect(contours[0])
                    x2,y2,w2,h2 = cv.boundingRect(contours[1])
                    current_variables= (min(x1,x2), min(y1,y2), max(x1+w1, x2+w2), max(y1+h1, y2 + h2))
            else:
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
            cv.imwrite('processed_images/'+image_name, image)
            if image_name in data:
                data[image_name]['aspect_ratio'] = image.shape[1]/image.shape[0]
    print('dumping data')
    with open('data.json', 'w') as output:
           json.dump(data, output, ensure_ascii=False, indent = 4)



def extract_features_for_training_data():
    data = dict()
    with open('data.json', 'r') as f:
        data = json.load(f)
        for i in data.keys():
            if not os.path.exists('processed_images/' + i):
                continue
            print('processed_images/' + i)

            image= cv.imread('processed_images/' + i)
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
        new_data = dict(data)
        for i in data.keys():
            if 'nb_of_pixels_per_segment' not in data[i]:
                new_data.pop(i)
    print('dumping data')
    with open('data.json', 'w') as output:
           json.dump(new_data, output, ensure_ascii=False, indent = 4)


def add_output_case():
    with open('data.json', 'r') as f:
        data = json.load(f)
        for i in data.keys():
            if data[i]['label'] == data[i]['label'].lower():
                data[i]['label_2'] = 'lower'
            else:
                data[i]['label_2'] = 'upper'
        with open('data.json', 'w') as output:
           json.dump(data, output, ensure_ascii=False, indent = 4)
# pre_process_images()

def get_total_nb_of_pixels():
        with open('data.json', 'r') as f:
            data = json.load(f)
            for i in data.keys():
                current_image = cv.imread('Img/' + i, 0)
                data[i]['nb_of_black_pixels'] = cv.countNonZero(current_image)  
            with open('data.json', 'w') as output:
                json.dump(data, output, ensure_ascii=False, indent = 4) 

# get_total_nb_of_pixels()
# convert_csv_to_json()
# pre_process_images()

extract_features_for_training_data()
# post_skeletonization()

# create_json()
# assign_random_colors()
# plot()

# add_output_case()