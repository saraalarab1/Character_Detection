
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
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeavePOut #for P-cross validation
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

sc = StandardScaler()


def train(X, Y, k_cross_validation_ratio, testing_size, optimal_k=True, max_range_k=100, model_version=None):
    
    X0_train, X_test, Y0_train, Y_test = train_test_split(X,Y,test_size=testing_size, random_state=7)
    #Scaler is needed to scale all the inputs to a similar range
    # scaler = StandardScaler()
    # scaler = scaler.fit(X0_train)
    # X0_train = scaler.transform(X0_train)
    # X_test = scaler.transform(X_test)
    #X_train, X_eval, Y_train, y_eval = train_test_split(X0_train, Y0_train, test_size= 100/k_cross_validation_ratio, random_state=7)
    

    # range for optimal K, can be specified by user
    if optimal_k and max_range_k>1:
        k_range= range(1, max_range_k)
    else:
        k_range=range(1,50)
    
    scores = {}
    scores_list = []

    # Set the parameters by cross-validation

    #finding the optimal nb of neighbors
    # print(k_range)
    for k in tqdm(k_range):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X0_train, Y0_train)
        y_pred = knn.predict(X_test)
        scores[k] = metrics.accuracy_score(Y_test, y_pred)
        scores_list.append(round(metrics.accuracy_score(Y_test, y_pred),3))
    print('all done')
    print(scores_list)
    k_optimal = scores_list.index(max(scores_list)) +1
    model = KNeighborsClassifier(n_neighbors= k_optimal)
    model.fit(X0_train, Y0_train)
     
    #eval_score_list = []
    #Evaluation using cross validation: lpo: leave p out
    #lpo = LeavePOut(p=1)
    accuracys=[]

    skf = StratifiedKFold(n_splits=10, random_state=None)
    skf.get_n_splits(X0_train, Y0_train)
    for train_index, test_index in skf.split(X0_train, Y0_train):
    
        # print("TRAIN:", train_index, "Validation:", test_index)
        X_train, X_eval = pd.DataFrame(X0_train).iloc[train_index], pd.DataFrame(X0_train).iloc[test_index]
        Y_train, y_eval = pd.DataFrame(Y0_train).iloc[train_index], pd.DataFrame(Y0_train).iloc[test_index]
    
        model.fit(X_train, Y_train)
        predictions = model.predict(X_eval)
        score = accuracy_score(predictions, y_eval)
        accuracys.append(score)
        #scores = cross_val_score(knn, X, Y, cv=5, scoring='accuracy')
        #eval_score_list.append(scores.mean())

    #eval_accuracy = np.mean(eval_score_list)
    eval_accuracy = np.mean(accuracys)

    #save the pretrained model:
    model_name='pretrained_knn_model.pkl'
    if model_version:
        pickle.dump(model, open(f"models/{model_version}/{model_name}", 'wb'))
    else:
        pickle.dump(model, open(f"models/knn/{model_name}", 'wb'))

    return eval_accuracy, model, X0_train, Y0_train, X_test, Y_test


def test(X_train, Y_train, X_test, Y_test,pretrain_model=True):

    if pretrain_model:
        model = pickle.load(open('models/knn/pretrained_knn_model.pkl', 'rb' ))
        
    else:
        eval_score, model, X_train, Y_train, X_test, Y_test = train(X_test, Y_test)
        print("Evaluation score: {}".format(eval_score))

    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    print("Text Prediction: {}".format(y_pred.shape))
    print("Y_test shape: {}".format(Y_test))
    classification_rep = classification_report(Y_test, y_pred,zero_division=True)
    test_score = metrics.accuracy_score(Y_test, y_pred)

    return test_score, classification_rep

def train_knn(features, model_version=None):
    print('training')
    x,y = get_input_output_labels(features)
    eval_accuracy, model, X_train, Y_train, X_test, Y_test = train(x, y, k_cross_validation_ratio=5, testing_size=0.05, max_range_k=100, model_version = model_version)
    test_score, conf_rep = test(X_train, Y_train, X_test, Y_test, pretrain_model=True)
    print("Evaluation Score: {}".format(eval_accuracy))
    print("Test Score: {}".format(test_score))
    print(conf_rep)
    return eval_accuracy, model, test_score, conf_rep
    
def get_input_output_labels(features):
    with open('data.json', 'r') as f: 
        data = json.load(f)
        x = []
        y = []
        for i in data.keys():
            for feature in features:
                x.append(data[i][feature])
            y.append(data[i]['label'])
    return (x,y)

train_knn(['vertical_histogram_projection'])