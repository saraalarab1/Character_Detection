
import json
import numpy as np 
import cv2
import yaml
import os
import pandas as pd
import pickle
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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

def train(X, Y, testing_size, model_version, for_ensemble, arabic= False):

    X0_train, X_test, Y0_train, Y_test = train_test_split(X,Y,test_size=testing_size, random_state=7)
    X0_train = X0_train[0:int(len(X0_train)/6)]
    Y0_train = Y0_train[0:int(len(Y0_train)/6)]
    scaling = MinMaxScaler(feature_range=(-1,1)).fit(X0_train)
    pickle.dump(scaling, open(f"models/{model_version}/scaler.pkl", 'wb'))
    X0_train = scaling.transform(X0_train)
    X_test = scaling.transform(X_test)
    #Scaler is needed to scale all the inputs to a similar range
    # scaler = StandardScaler()
    # scaler = scaler.fit(X0_train)
    # X0_train = scaler.transform(X0_train)
    # X_test = scaler.transform(X_test)
    
    model = SVC(kernel = 'rbf', random_state = 0, probability= True)
    model.fit(X0_train, Y0_train)
    print('finished initial training')
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

    eval_accuracy = np.mean(accuracys)

    #save the pretrained model:
    model_name = "pretrained_svm_model.pkl"
    if arabic: 
        model_name = "pretrained_svm_arabic_model.pkl"
    if model_version:
        pickle.dump(model, open(f"models/{model_version}/{model_name}", 'wb'))
    elif for_ensemble:
        pickle.dump(model, open(f"models/svm_ensemble/{model_name}", 'wb'))
    else:
        pickle.dump(model, open(f"models/svm/{model_name}", 'wb'))

    return eval_accuracy, model, X_test, Y_test

def test(X_test, Y_test,model_version,for_ensemble, arabic= False):
    model_name = "pretrained_svm_model.pkl"
    if arabic: 
        model_name = "pretrained_svm_arabic_model.pkl"
    if model_version:
        model = pickle.load(open(f'models/{model_version}/{model_name}', 'rb' ))
    elif for_ensemble:
        model = pickle.load(open('models/svm_ensemble/{model_name}', 'rb' ))
    else:
        model = pickle.load(open('models/svm/{model_name}', 'rb' ))

    y_pred = model.predict(X_test)
    print(y_pred)
    classification_rep = classification_report(Y_test, y_pred, zero_division=True)
    test_score = metrics.accuracy_score(Y_test, y_pred)

    return test_score, classification_rep

def train_svm(features, model_version=None, for_ensemble = False, arabic = False):
    print('training')
    x,y = get_input_output_labels(features, arabic)
    eval_accuracy, model, X_test, Y_test = train(x, y, testing_size=0.2, model_version = model_version, for_ensemble = for_ensemble, arabic=arabic)
    test_score, conf_rep = test(X_test, Y_test,model_version=model_version, for_ensemble=for_ensemble, arabic= arabic)
    print(conf_rep)
    print("Evaluation Score: {}".format(eval_accuracy))
    print("Test Score: {}".format(test_score))
    if model_version is None and not for_ensemble:
        save_model(eval_accuracy, test_score, conf_rep ,features,arabic)
    return eval_accuracy, model, test_score, conf_rep

def get_input_output_labels(features, arabic):
    data_file = 'data.json'
    if arabic:
        data_file = 'arabic_data.json'
    with open(data_file, 'r') as f: 
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

def save_model(eval_accuracy, test_score, conf_rep, features, arabic):
    yaml_info = dict()
    model_language = 'english'
    if arabic:
        model_language = 'arabic'
    yaml_info['prediction_model'] =  [f'pretrained_svm_${model_language}model.pkl']
    yaml_info['features'] = features
    yaml_info['training'] = 'completed'
    yaml_info['name'] = 'svm'
    yaml_info['eval_accuracy'] = float(eval_accuracy)
    yaml_info['test_score'] = float(test_score)
    yaml_info['weight'] = 1
    yaml_info['conf_rep'] = get_info(conf_rep)

    model_version="svm"
    yaml_path = os.path.join("models",model_version, 'model.yaml')
    with open(yaml_path, 'w') as output:
        yaml.dump(yaml_info, output)

def get_info(conf_rep):
    print(conf_rep)
    data = conf_rep.splitlines()[2:len(conf_rep)-5]
    print(data)
    # average_data =  conf_rep.splitlines()[62:65]
    # average_data = " ".join(label_information.split())
    # print(average_data)
    label_data = []
    for label_information in data:
        print('current_info', label_information)

        if len(label_information) < 4:
            break
        label_information = " ".join(label_information.split())
        label_information = label_information.split(" ")
        label_data.append({label_information[0]:[float(label_information[1]),float(label_information[2]),float(label_information[3])]})

    return label_data

# train_svm(['nb_of_pixels_per_segment','horizontal_line_intersection','vertical_line_intersection'],for_ensemble=False)