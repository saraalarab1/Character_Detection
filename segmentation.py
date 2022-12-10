import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
from skeletonize import skeletonize_image
import more_itertools as mit
from pre_processing import gray_to_black, convert_dark_to_black_and_light_to_white
from paragraph_segmentation import paragraph_seg
image = cv.imread('predictions/current2/image_2.png')
import cv2

def find_ranges(iterable, type):
    """Yield and filter middle of consecutive numbers."""
    if type == 0: 
        ran = 5
    else:
        ran = 100
    for group in mit.consecutive_groups(iterable):
        group = list(group)
        if group.__contains__(0):
            continue
        if len(group) <ran :
            continue
        else:
            yield int((group[0]+ group[-1])/2)

import cv2

def resize_image(image, new_height):
  # Check the current width and height of the image
  height, width = image.shape[:2]

  # Calculate the new width, maintaining the aspect ratio
  new_width = int(width * (new_height / height))

  # Resize the image
  resized_image = cv2.resize(image, (new_width, new_height))

  return resized_image

def divide_image(columns, image):
  sub_images = []
  columns = [0] + columns
  for i in range(len(columns) - 1):
    sub_image = image[:, columns[i]:columns[i+1]]

    sub_images.append(sub_image)

  return sub_images


import cv2
import numpy as np

def word_segmentation(image):
    image = resize_image(image, new_height= 100)
    # Finding Contours
    # Use a copy of the image e.g. edged.copy()
    # since findContours alters the image
    old_image = image.copy()
    image = convert_dark_to_black_and_light_to_white(image)
    cv.imshow('image converted', image)
    cv.waitKey(0)
    image = skeletonize_image(image, 'image')
    transposed_image = cv.transpose(image)

    onePixelsColumnsIndices= []
    zeroPixelsColumnsIndices = []
    rows, cols, _ = transposed_image.shape
    startingPixels = []
    endingPixels = []

    for i in range(rows):
        current_starting_pixel = 0
        current_ending_pixels = 0
        for j in range(cols):
            k = transposed_image[i, j]
            if tuple(k)!= (255, 255, 255):
                if current_starting_pixel ==0:
                    current_starting_pixel = j
                current_ending_pixels = j
        startingPixels.append(current_starting_pixel)
        endingPixels.append(current_ending_pixels)
    avg_starting_pixels = int(sum(startingPixels) / len(startingPixels))
    avg_ending_pixels = int(sum(endingPixels) /len(endingPixels))   
    med =  int((avg_starting_pixels + avg_ending_pixels) /2)
    for i in range(rows):
        nbOfBlackPixelsPerRow = 0
        current_col = 0
        for j in range(cols):
            k = transposed_image[i, j]
            if tuple(k) != (255, 255, 255):
                nbOfBlackPixelsPerRow = nbOfBlackPixelsPerRow + 1
                current_col = j
        if nbOfBlackPixelsPerRow == 0:
            zeroPixelsColumnsIndices.append(i)
        if nbOfBlackPixelsPerRow == 1 and current_col > med:
            onePixelsColumnsIndices.append(i)    

    seperation = list(find_ranges(onePixelsColumnsIndices, 1))
    seperation.extend(list(find_ranges(zeroPixelsColumnsIndices, 0)))   
    seperation.sort() 
    rows, cols, _ = image.shape
    letters = divide_image(seperation, old_image)
    for letter in letters:
        if letter.shape[1] >16:
            cv.imshow('image', letter)
            cv.waitKey(0)
    return letters

image = cv2.imread('segmentation_hello.png')
word_segmentation(image)