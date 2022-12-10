import cv2
import numpy as np
import matplotlib.pyplot as plt
import itertools

img = cv2.imread('newimage.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
def thresholding(image, folder_path):
    img_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(img_gray,80,255,cv2.THRESH_BINARY_INV)
    cv2.imwrite(folder_path+'image.png', thresh)
    return thresh

def get_area(word):
    return (word[2] - word[0]) * (word[3] - word[1])

def paragraph_seg(img, folder_path):
    h, w, c = img.shape

    if w > 1000:
        
        new_w = 1000
        ar = w/h
        new_h = int(new_w/ar)
        
        img = cv2.resize(img, (new_w, new_h), interpolation = cv2.INTER_AREA)
    cv2.imwrite(folder_path+'image_resized.png', img)


    thresh_img = thresholding(img, folder_path);


    #dilation
    kernel = np.ones((3,85), np.uint8)
    dilated = cv2.dilate(thresh_img, kernel, iterations = 1)
    cv2.imwrite(folder_path+'image_dilated.png', dilated)
    (contours, heirarchy) = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_contours_lines = sorted(contours, key = lambda ctr : cv2.boundingRect(ctr)[1]) # (x, y, w, h)


    img2 = img.copy()

    for ctr in sorted_contours_lines:
        
        x,y,w,h = cv2.boundingRect(ctr)
        cv2.rectangle(img2, (x,y), (x+w, y+h), (40, 100, 250), 2)
        
    cv2.imwrite(folder_path+'image_countoured.png', img2)

    #dilation
    kernel = np.ones((3,15), np.uint8)
    dilated2 = cv2.dilate(thresh_img, kernel, iterations = 1)
    cv2.imwrite(folder_path+'image_dilated2.png', dilated2)
    img3 = img.copy()
    words_list = []

    for line in sorted_contours_lines:
        
        # roi of each line
        x, y, w, h = cv2.boundingRect(line)
        roi_line = dilated2[y:y+w, x:x+w]
        
        # draw contours on each word
        (cnt, heirarchy) = cv2.findContours(roi_line.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        sorted_contour_words = sorted(cnt, key=lambda cntr : cv2.boundingRect(cntr)[0])
        
        for word in sorted_contour_words:
            
            if cv2.contourArea(word) < 400:
                continue
            
            x2, y2, w2, h2 = cv2.boundingRect(word)
            if [x+x2, y+y2, x+x2+w2, y+y2+h2] not in words_list:
                words_list.append([x+x2, y+y2, x+x2+w2, y+y2+h2])
            cv2.rectangle(img3, (x+x2, y+y2), (x+x2+w2, y+y2+h2), (255,255,100),2)
            
    cv2.imwrite(folder_path+'image_word_contoured.png', img3)
    words_list.sort()
    new_words_list = []
    words_list = list(l for l, _ in itertools.groupby(words_list))
    words = []
    for i in range(len(words_list)):
        current_word = words_list[i]
        current_image = img[current_word[1]:current_word[3], current_word[0]:current_word[2]]
        words.append(current_image)
        cv2.imwrite(folder_path+'image_'+str(i)+'.png', current_image)
    return words
