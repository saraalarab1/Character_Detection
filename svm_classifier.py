
import json
import numpy as np 
import cv2
import pandas as pd
import pickle
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeavePOut #for P-cross validation
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

def get_gamma_and_C(model):
    # creating a KFold object with 5 splits 
    folds = KFold(n_splits = 5, shuffle = True, random_state = 101)

    #specify range of hyperparameters
    hyper_params = [ {'gamma': [1e-2, 1e-3, 1e-4],
                        'C': [1, 10, 100, 1000]}]

    model = GridSearchCV(
        estimator= model, 
        param_grid= hyper_params,
        scoring= 'accuracy',
        cv = folds, 
        verbose= 1,
        return_train_score= True)

    return model.best_params_

def train(X, Y, k_cross_validation_ratio, testing_size, model_version=None):

    X0_train, X_test, Y0_train, Y_test = train_test_split(X,Y,test_size=testing_size, random_state=7)
    #Scaler is needed to scale all the inputs to a similar range
    # scaler = StandardScaler()
    # scaler = scaler.fit(X0_train)
    # X0_train = scaler.transform(X0_train)
    # X_test = scaler.transform(X_test)
    #X_train, X_eval, Y_train, y_eval = train_test_split(X0_train, Y0_train, test_size= 100/k_cross_validation_ratio, random_state=7)
    
    model = SVC(kernel = 'rbf', random_state = 0, probability= True)
    model.fit(X0_train, Y0_train)

    #eval_score_list = []
    #Evaluation using cross validation: lpo: leave p out
    # lpo = LeavePOut(p=1)

    accuracys=[]

    skf = StratifiedKFold(n_splits=10, random_state=None)
    skf.get_n_splits(X0_train, Y0_train)
    for train_index, test_index in skf.split(X0_train, Y0_train):
    
        # print("TRAIN:", train_index, "Validation:", test_index)
        X_train, X_eval = pd.DataFrame(X0_train).iloc[train_index], pd.DataFrame(X0_train).iloc[test_index]
        Y_train, y_eval = pd.DataFrame(Y0_train).iloc[train_index], pd.DataFrame(Y0_train).iloc[test_index]

        model.fit(X_train, Y_train.values.ravel())
        predictions = model.predict(X_eval)
        score = accuracy_score(predictions, y_eval)
        accuracys.append(score)
        #scores = cross_val_score(knn, X, Y, cv=5, scoring='accuracy')
        #eval_score_list.append(scores.mean())

    #eval_accuracy = np.mean(eval_score_list)
    eval_accuracy = np.mean(accuracys)

    #save the pretrained model:
    model_name = "pretrained_svm_model.pkl"
    if model_version:
        pickle.dump(model, open(f"models/{model_version}/{model_name}", 'wb'))
    else:
        pickle.dump(model, open(f"models/svm/{model_name}", 'wb'))

    return eval_accuracy, model, X_test, Y_test

def test(X_test, Y_test,model_version):

    if model_version:
        model = pickle.load(open(f'models/{model_version}/pretrained_svm_model.pkl', 'rb' ))
    else:
        model = pickle.load(open('models/svm/pretrained_svm_model.pkl', 'rb' ))

    y_pred = model.predict(X_test)
    print("Text Prediction: {}".format(y_pred.shape))
    print("Y_test shape: {}".format(Y_test))
    classification_rep = classification_report(Y_test, y_pred, zero_division=True)
    test_score = metrics.accuracy_score(Y_test, y_pred)

    return test_score, classification_rep

def train_svm(features, model_version=None):
    print('training')
    x,y = get_input_output_labels(features)
    eval_accuracy, model, X_test, Y_test = train(x, y, k_cross_validation_ratio=5, testing_size=0.05, model_version = model_version)
    test_score, conf_rep = test(X_test, Y_test,model_version=model_version)
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
                features_arr = []
                for feature in features:
                    arr = data[i][feature]
                    if type(arr) != list:
                        arr = [arr]
                    features_arr.extend(arr)
                x.append(features_arr)
            y.append(data[i]['label'])
    return (x,y)

train_svm(['nb_of_pixels_per_segment'])
