
import json
import numpy as np 
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import os
# from skimage import feature
from tqdm import tqdm
import glob
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeavePOut #for P-cross validation
from sklearn.metrics import classification_report, accuracy_score


def train(X, y, k_cross_validation_ratio, testing_size, optimal_k=True, max_range_k=0 ):

    X0_train, X_test, y0_train, y_test = train_test_split(X,y,test_size=testing_size, random_state=7)
    #Scaler is needed to scale all the inputs to a similar range
    # scaler = StandardScaler()
    # scaler = scaler.fit(X0_train)
    # X0_train = scaler.transform(X0_train)
    # X_test = scaler.transform(X_test)
    #X_train, X_eval, y_train, y_eval = train_test_split(X0_train, y0_train, test_size= 100/k_cross_validation_ratio, random_state=7)
    

    #finding the range for the optimal value of k either within the specified range (user input) 
    # or by our default range
    if optimal_k and max_range_k>1:
        k_range= range(1, max_range_k)
    else:
        k_range=range(1,50)
    

    scores = {}
    scores_list = []

    #finding the optimal nb of neighbors
    print(k_range)
    for k in tqdm(k_range):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X0_train, y0_train)
        y_pred = knn.predict(X_test)
        scores[k] = metrics.accuracy_score(y_test, y_pred)
        scores_list.append(metrics.accuracy_score(y_test, y_pred))
    print(scores)
    k_optimal = scores_list.index(max(scores_list))
    model = KNeighborsClassifier(n_neighbors= k_optimal)
    print(k_optimal)
    return 

    eval_score_list = []
    #Evaluation using cross validation: lpo: leave p out
    from sklearn.model_selection import StratifiedKFold
    lpo = LeavePOut(p=1)
    accuracys=[]

    skf = StratifiedKFold(n_splits=10, random_state=None)
    skf.get_n_splits(X0_train, y0_train)
    for train_index, test_index in skf.split(X0_train, y0_train):
    
        # print("TRAIN:", train_index, "Validation:", test_index)
        X_train, X_eval = pd.DataFrame(X0_train).iloc[train_index], pd.DataFrame(X0_train).iloc[test_index]
        y_train, y_eval = pd.DataFrame(y0_train).iloc[train_index], pd.DataFrame(y0_train).iloc[test_index]
    
        model.fit(X0_train, y0_train)
        predictions = model.predict(X_eval)
        score = accuracy_score(predictions, y_eval)
        accuracys.append(score)
        #scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
        #eval_score_list.append(scores.mean())

    #eval_accuracy = np.mean(eval_score_list)
    eval_accuracy = np.mean(accuracys)

    #save the pretrained model:
    model_name='pretrained_knn_model'
    pickle.dump(model, open(model_name, 'wb'))

    return eval_accuracy, model, X0_train, y0_train, X_test, y_test


def test(X_train, y_train, X_test, y_test,pretrain_model=False):
    model_name='pretrained_knn_model'
    if pretrain_model:
        model = pickle.load(open(model_name, 'rb' ))
        
    else:
        eval_score, model, X_train, y_train, X_test, y_test = train(X_test, y_test, pretrained_model=False)
        print("Evaluation score: {}".format(eval_score))

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Predictions shape: {}".format(y_pred.shape))
    print("Y_test shape: {}".format(y_test))
    classification_rep = classification_report(y_test, y_pred)
    test_score = metrics.accuracy_score(y_test, y_pred)

    return test_score, classification_rep


with open('data_with_colors.json', 'r') as f: 
    data = json.load(f)
    x = []
    y = []
    for i in data.keys():
        x.append([data[i]['vertical_ratio'], data[i]['horizontal_ratio'], data[i]['aspect_ratio'], data[i]['percentage_of_pixels_at_horizontal_center'], data[i]['percentage_of_pixels_at_vertical_center'], data[i]['horizontal_line_intersection_count'], data[i]['vertical_line_intersection_count']])
        y.append(data[i]['label'])

eval_accuracy, model, X_train, y_train, X_test, y_test = train(x, y, k_cross_validation_ratio=5, testing_size=0.2, max_range_k=100)
# test_score, conf_rep = test(X_train, y_train,X_test, y_test, pretrain_model=True)
# print("Evaluation Score: {}".format(eval_accuracy))
# print("Test Score: {}".format(test_score))
# print(conf_rep)