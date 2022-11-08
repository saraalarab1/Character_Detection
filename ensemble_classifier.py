import pickle
from datetime import datetime
from genericpath import exists
from os import makedirs
import os
from flask import Flask, jsonify, render_template, request, redirect, url_for
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from matplotlib.style import available
import numpy as np
import json
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import VotingClassifier

import pandas as pd
import matplotlib.pyplot as plt
import os
# from skimage import feature

import pickle
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn import metrics
from sklearn import model_selection
from sklearn.model_selection import LeavePOut #for P-cross validation
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, plot_confusion_matrix

def train(X, y, estimators, weights, model_version, testing_size):

    X0_train, X_test, Y0_train, y_test = train_test_split(X,y,test_size=testing_size, random_state=7)

    ensemble=VotingClassifier(estimators=estimators, voting='soft', weights=weights)
    ensemble.fit(X0_train, Y0_train)
    
    Y_pred = ensemble.predict(X_test)
    eval_accuracy = accuracy_score(Y_pred, y_test)

    cm = confusion_matrix(Y_pred, y_test)
    labels=['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']   
    cmd = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labels)
    cmd.plot()

    #save the pretrained model:
    model_name='pretrained_ensemble_model.pkl'
    if model_version:
        pickle.dump(ensemble, open(f"models/{model_version}/{model_name}", 'wb'))
    else:
        pickle.dump(ensemble, open(f"models/ensemble/{model_name}", 'wb'))

    return eval_accuracy, ensemble, X0_train, Y0_train, X_test, y_test

def test(X_train, Y_train, X_test, Y_test,model_version):

    if model_version:
        model = pickle.load(open(f'models/{model_version}/pretrained_ensemble_model.pkl', 'rb' ))
    else:
        model = pickle.load(open(f'models/ensemble/pretrained_ensemble_model.pkl', 'rb' ))
        
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    confusion_matrix(y_pred, Y_test)
    plot_confusion_matrix(model, X_test, Y_test, cmap=plt.cm.Blues)
    plt.show()
    classification_rep = classification_report(Y_test, y_pred,zero_division=True)
    test_score = metrics.accuracy_score(Y_test, y_pred)

    return test_score, classification_rep

def train_ensemble(estimators, weights,features, model_version=None):
    print('training')
    X,y = get_input_output_labels(features)
    eval_accuracy, model, X_train, Y_train, X_test, Y_test = train(X, y, estimators, weights,model_version= model_version, testing_size=0.1,)
    test_score, conf_rep = test(X_train, Y_train, X_test, Y_test, model_version)
    print(conf_rep)
    print("Evaluation Score: {}".format(eval_accuracy))
    print("Test Score: {}".format(test_score))
    return eval_accuracy, model, test_score, conf_rep


def get_input_output_labels(features):
    with open('data.json', 'r') as f: 
        data = json.load(f)
        x = []
        y = []
        for i in data.keys():
            for feature in features:
                features_arr = []
                for feature in features:
                    arr = data[i][feature]
                    if type(arr) != list:
                        arr = [arr]
                    features_arr.extend(arr)
            x.append(features_arr)
            y.append(data[i]['label'])
    return (x,y)

knn = pickle.load(open(f'models/knn_ensemble/pretrained_knn_model.pkl', 'rb' ))
svm = pickle.load(open(f'models/svm_ensemble/pretrained_svm_model.pkl', 'rb' ))

# train_ensemble([('KNN',knn),('SVM',svm)],[1,1],['nb_of_pixels_per_segment','horizontal_line_intersection','vertical_line_intersection'])
