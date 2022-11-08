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
import yaml
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
    # confusion_matrix(y_pred, Y_test)
    # plot_confusion_matrix(model, X_test, Y_test, cmap=plt.cm.Blues)
    # plt.show()
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
    if model_version is None:
        save_model(eval_accuracy, test_score, conf_rep ,features)
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

def save_model(eval_accuracy, test_score, conf_rep, features ):
    yaml_info = dict()

    yaml_info['prediction_model'] = "pretrained_ensemble_model.pkl"
    yaml_info['features'] = features
    yaml_info['training'] = 'completed'
    yaml_info['name'] = 'pretrained_ensemble_model.pkl'

    model_version="ensemble"

    yaml_path = os.path.join("models",model_version, 'model.yaml')
    with open(yaml_path, 'w') as output:
        yaml.dump(yaml_info, output)

        yaml_info['ensemble'] = dict()
        yaml_info['ensemble']['eval_accuracy'] = float(eval_accuracy)
        yaml_info['ensemble']['test_score'] = float(test_score)
        yaml_info['ensemble']['conf_rep'] = get_info(conf_rep)
        yaml_info['ensemble']['weight'] = 1

    with open(yaml_path, 'w') as output:
        yaml.dump(yaml_info, output)

def get_info(conf_rep):
    data = conf_rep.splitlines()[2:61]
    # average_data =  conf_rep.splitlines()[62:65]
    # average_data = " ".join(label_information.split())
    # print(average_data)
    label_data = []
    for label_information in data:
        label_information = " ".join(label_information.split())
        label_information = label_information.split(" ")
        if len(label_information) < 4:
            continue
        label_data.append({label_information[0]:[float(label_information[1]),float(label_information[2]),float(label_information[3])]})

    return label_data

# knn = pickle.load(open(f'models/knn_ensemble/pretrained_knn_model.pkl', 'rb' ))
# svm = pickle.load(open(f'models/svm_ensemble/pretrained_svm_model.pkl', 'rb' ))

# train_ensemble([('KNN',knn),('SVM',svm)],[1,1],['nb_of_pixels_per_segment','horizontal_line_intersection','vertical_line_intersection'])
