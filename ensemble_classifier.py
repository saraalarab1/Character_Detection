import pickle
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

def train(x, y, estimators, weights, testing_size, model_version):

    X0_train, X_test, Y0_train, y_test = train_test_split(x,y,test_size=testing_size, random_state=7)

    ensemble=VotingClassifier(estimators=estimators, voting='soft', weights=weights)
    ensemble.fit(X0_train, Y0_train)

    accuracys=[]

    skf = StratifiedKFold(n_splits=10, random_state=None)
    skf.get_n_splits(X0_train, Y0_train)
    for train_index, test_index in skf.split(X0_train, Y0_train):
        # print("TRAIN:", train_index, "Validation:", test_index)
        X_train, X_eval = pd.DataFrame(X0_train).iloc[train_index], pd.DataFrame(X0_train).iloc[test_index]
        Y_train, y_eval = pd.DataFrame(Y0_train).iloc[train_index], pd.DataFrame(Y0_train).iloc[test_index]
    
        ensemble.fit(X_train, Y_train.values.ravel())
        predictions = ensemble.predict(X_eval)
        score = accuracy_score(predictions, y_eval)
        accuracys.append(score)

    eval_accuracy = np.mean(accuracys)

    #save the pretrained model:
    model_name='pretrained_ensemble_model.pkl'
    if model_version:
        pickle.dump(ensemble, open(f"models/{model_version}/{model_name}", 'wb'))
    else:
        pickle.dump(ensemble, open(f"models/ensemble/{model_name}", 'wb'))

    return eval_accuracy, ensemble, X_test, y_test

def test(X_test, Y_test,model_version):

    if model_version:
        model = pickle.load(open(f'models/{model_version}/pretrained_ensemble_model.pkl', 'rb' ))
    else:
        model = pickle.load(open('models/ensemble/pretrained_ensemble_model.pkl', 'rb' ))

    y_pred = model.predict(X_test)
    print("Text Prediction: {}".format(y_pred.shape))
    print("Y_test shape: {}".format(Y_test))
    classification_rep = classification_report(Y_test, y_pred,zero_division=True)
    test_score = metrics.accuracy_score(Y_test, y_pred)

    return test_score, classification_rep

def train_ensemble(estimators, weights,features, model_version=None):
    print('training')
    x,y = get_input_output_labels(features)
    eval_accuracy, model, X_test, Y_test = train(x, y, estimators, weights, testing_size=0.1, model_version= model_version)
    test_score, conf_rep = test(X_test, Y_test, model_version)
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


knn = pickle.load(open(f'models/knn/pretrained_knn_model.pkl', 'rb' ))
svm = pickle.load(open(f'models/svm/pretrained_svm_model.pkl', 'rb' ))
dtree = pickle.load(open(f'models/dt/pretrained_dtree_model.pkl', 'rb' ))

# train_ensemble([('KNN',knn),('SVM',svm),('DTree',dtree)],[1,1,1],['nb_of_pixels_per_segment'])
train_ensemble([('KNN',knn),('SVM',svm)],[1,1],['nb_of_pixels_per_segment'])
