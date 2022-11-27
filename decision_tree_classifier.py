
import json
import numpy as np 
import pickle
from pandas import DataFrame as df
from sklearn import metrics
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import StratifiedKFold, KFold, LeavePOut #for P-cross validation
from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import os
import yaml

def train(x, y, testing_size, model_version,for_ensemble, arabic):

    X0_train, X_test, Y0_train, Y_test = train_test_split(x,y,test_size=testing_size, random_state=7)
    #Scaler is needed to scale all the inputs to a similar range
    # scaler = StandardScaler()
    # scaler = scaler.fit(X0_train)
    # X0_train = scaler.transform(X0_train)
    # X_test = scaler.transform(X_test)
    #X_train, X_eval, Y_train, y_eval = train_test_split(X0_train, Y0_train, test_size= 100/k_cross_validation_ratio, random_state=7)
    
    model = DecisionTreeClassifier()
    model.fit(X0_train, Y0_train)

    accuracy = cross_val_score(model, X0_train, Y0_train, cv=5, scoring='accuracy')
    print(f"{accuracy}")

    accuracys=[]

    skf = StratifiedKFold(n_splits=5, random_state=None)
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
    model_language = 'english'
    if arabic: 
        model_language = 'arabic'
    #save the pretrained model:
    model_name=f'pretrained_dtree_{model_language}_model.pkl'
    if model_version:
        pickle.dump(model, open(f"models/{model_version}/{model_name}", 'wb'))
    elif for_ensemble:
        pickle.dump(model, open(f"models/d_tree_ensemble/{model_name}", 'wb'))
    else:
        pickle.dump(model, open(f"models/d_tree/{model_name}", 'wb'))

    return eval_accuracy, model, X_test, Y_test


def test(X_test, Y_test, model_version,for_ensemble, arabic):
    model_language = 'english'
    if arabic: 
        model_language = 'arabic'
    if model_version: 
        model = pickle.load(open(f'models/{model_version}/pretrained_dtree_{model_language}_model.pkl', 'rb' ))
    elif for_ensemble:
        model = pickle.load(open('modelsd_tree_ensemble/pretrained_dtree_{model_language}_model.pkl', 'rb' ))
    else:
        model = pickle.load(open('models/d_tree/pretrained_dtree_{model_language}_model.pkl', 'rb' ))

    y_pred = model.predict(X_test)
    print("Text Prediction: {}".format(y_pred.shape))
    print("Y_test shape: {}".format(Y_test))
    classification_rep = classification_report(Y_test, y_pred,zero_division=True)
    test_score = metrics.accuracy_score(Y_test, y_pred)

    return test_score, classification_rep

def train_dt(features, model_version=None, for_ensemble = False, arabic =False):
    print('training')
    x,y = get_input_output_labels(features, arabic)
    eval_accuracy, model, X_test, Y_test = train(x, y, testing_size=0.05, model_version = model_version , for_ensemble=for_ensemble, arabic= arabic)
    test_score, conf_rep = test(X_test, Y_test, model_version = model_version,for_ensemble=for_ensemble, arabic= arabic)
    print("Evaluation Score: {}".format(eval_accuracy))
    print("Test Score: {}".format(test_score))
    print(conf_rep)
    if model_version is None:
        save_model(eval_accuracy, test_score, conf_rep,for_ensemble ,features, arabic)
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

def save_model(eval_accuracy, test_score, conf_rep, for_ensemble, features, arabic):
    model_language = 'english'
    if arabic: 
        model_language = 'arabic'
    yaml_info = dict()

    yaml_info['prediction_model'] = "pretrained_dtree_{model_language}_model.pkl"
    yaml_info['features'] = features
    yaml_info['training'] = 'completed'
    yaml_info['name'] = 'pretrained_dtree_{model_language}_model.pkl'

    model_version="d_tree"
    if for_ensemble:
        model_version = "d_tree_ensemble"

    yaml_path = os.path.join("models",model_version, 'model.yaml')
    with open(yaml_path, 'w') as output:
        yaml.dump(yaml_info, output)

        yaml_info['d_tree'] = dict()
        yaml_info['d_tree']['eval_accuracy'] = float(eval_accuracy)
        yaml_info['d_tree']['test_score'] = float(test_score)
        yaml_info['d_tree']['conf_rep'] = get_info(conf_rep)
        yaml_info['d_tree']['weight'] = 1

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
            break
        label_data.append({label_information[0]:[float(label_information[1]),float(label_information[2]),float(label_information[3])]})

    return label_data

# train_dt(['nb_of_pixels_per_segment'])
