
import json
import numpy as np 
import pickle
from pandas import DataFrame as df
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.model_selection import  train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import os
import yaml

secondLayerLetters = 'uUvVzZxXkKjJnNmM0OoPpSsCcYy'

def get_gamma_and_C(model, X_train, Y_train):
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
    model.fit(X_train, Y_train)    
    return model.best_params_

def train(x, y, testing_size, model_version):

    X0_train, X_test, Y0_train, Y_test = train_test_split(x,y,test_size=testing_size, random_state=7)
    #Scaler is needed to scale all the inputs to a similar range
    # scaler = StandardScaler()
    # scaler = scaler.fit(X0_train)
    # X0_train = scaler.transform(X0_train)
    # X_test = scaler.transform(X_test)
    
    model = SVC(kernel = 'rbf', random_state = 0, probability= True)
    model.fit(X0_train, Y0_train)

    accuracys=[]

    # StratifiedKFold
    skf = StratifiedKFold(n_splits=10, random_state=None)
    skf.get_n_splits(X0_train, Y0_train)
    for train_index, test_index in skf.split(X0_train, Y0_train):
    
        # print("TRAIN:", train_index, "Validation:", test_index)
        X_train, X_eval = df(X0_train).iloc[train_index], df(X0_train).iloc[test_index]
        Y_train, y_eval = df(Y0_train).iloc[train_index], df(Y0_train).iloc[test_index]

        model.fit(X_train, Y_train.values.ravel())
        predictions = model.predict(X_eval)
        score = accuracy_score(predictions, y_eval)
        accuracys.append(score)

    eval_accuracy = np.mean(accuracys)

    #save the pretrained model:
    model_name = "pretrained_svm_model.pkl"
    if model_version:
        pickle.dump(model, open(f"models/{model_version}/{model_name}", 'wb'))
    else:
        pickle.dump(model, open(f"models/svm_case/{model_name}", 'wb'))

    return eval_accuracy, model, X_test, Y_test

def test(X_test, Y_test,model_version):

    if model_version:
        model = pickle.load(open(f'models/{model_version}/pretrained_svm_model.pkl', 'rb' ))
    else:
        model = pickle.load(open('models/svm_case/pretrained_svm_model.pkl', 'rb' ))

    y_pred = model.predict(X_test)
    classification_rep = classification_report(Y_test, y_pred, zero_division=True)
    test_score = metrics.accuracy_score(Y_test, y_pred)

    return test_score, classification_rep

def train_svm_case(features, model_version=None):
    print('training')
    x,y = get_input_output_labels(features)
    eval_accuracy, model, X_test, Y_test = train(x, y, testing_size=0.2, model_version = model_version)
    test_score, conf_rep = test(X_test, Y_test,model_version=model_version)
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
            if data[i]['label'] in secondLayerLetters:
                for feature in features:
                    features_arr = []
                    for feature in features:
                        arr = data[i][feature]
                        if type(arr) != list:
                            arr = [arr]
                        features_arr.extend(arr)
                x.append(features_arr)
                y.append(data[i]['label_2'])
    return (x,y)

def save_model(eval_accuracy, test_score, conf_rep, features ):
    yaml_info = dict()

    yaml_info['prediction_model'] = "pretrained_svm_model.pkl"
    yaml_info['features'] = features
    yaml_info['training'] = 'completed'
    yaml_info['name'] = 'svm_case'
    yaml_info['eval_accuracy'] = float(eval_accuracy)
    yaml_info['test_score'] = float(test_score)
    yaml_info['weight'] = 1
    yaml_info['conf_rep'] = get_info(conf_rep)

    model_version="svm_case"
    yaml_path = os.path.join("models",model_version, 'model.yaml')
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
        label_data.append({label_information[0]:[label_information[1],label_information[2],label_information[3]]})

    return label_data

# train_svm(['nb_of_pixels_per_segment','aspect_ratio'])
