
import json
import yaml
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


def train(X, Y, testing_size, for_ensemble,model_version, optimal_k=True, max_range_k=100, ):
    
    X0_train, X_test, Y0_train, Y_test = train_test_split(X,Y,test_size=testing_size, random_state=7)
    #Scaler is needed to scale all the inputs to a similar range
    # scaler = StandardScaler()
    # scaler = scaler.fit(X0_train)
    # X0_train = scaler.transform(X0_train)
    # X_test = scaler.transform(X_test)
    
    # range for optimal K, can be specified by user
    if optimal_k and max_range_k>1:
        k_range= range(1, max_range_k)
    else:
        k_range=range(1,50)
    
    scores = {}
    scores_list = []

    #finding the optimal nb of neighbors
    for k in tqdm(k_range):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X0_train, Y0_train)
        y_pred = knn.predict(X_test)
        scores[k] = metrics.accuracy_score(Y_test, y_pred)
        scores_list.append(round(metrics.accuracy_score(Y_test, y_pred),3))
    k_optimal = scores_list.index(max(scores_list)) +1
    print(f"k optimal: {k_optimal}")
    model = KNeighborsClassifier(n_neighbors= k_optimal)
    model.fit(X0_train, Y0_train)
     
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
    model_name='pretrained_knn_model.pkl'
    if model_version:
        pickle.dump(model, open(f"models/{model_version}/{model_name}", 'wb'))
    elif for_ensemble:
        pickle.dump(model, open(f"models/knn_ensemble/{model_name}", 'wb'))
    else:
        pickle.dump(model, open(f"models/knn/{model_name}", 'wb'))

    return eval_accuracy, model, X0_train, Y0_train, X_test, Y_test


def test(X_train, Y_train, X_test, Y_test, model_version, for_ensemble):

    if model_version: 
        model = pickle.load(open(f'models/{model_version}/pretrained_knn_model.pkl', 'rb' ))
    elif for_ensemble:
        model = pickle.load(open('models/knn_ensemble/pretrained_knn_model.pkl', 'rb' ))
    else:
        model = pickle.load(open('models/knn/pretrained_knn_model.pkl', 'rb' ))
        
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    classification_rep = classification_report(Y_test, y_pred,zero_division=True)
    test_score = metrics.accuracy_score(Y_test, y_pred)

    return test_score, classification_rep

def train_knn(features, model_version=None, for_ensemble = False):
    print('training')
    x,y = get_input_output_labels(features)
    eval_accuracy, model, X_train, Y_train, X_test, Y_test = train(x, y, testing_size=0.2, max_range_k=50, model_version = model_version, for_ensemble=for_ensemble)
    test_score, conf_rep = test(X_train, Y_train, X_test, Y_test, model_version=model_version,for_ensemble = for_ensemble)
    print(conf_rep)
    print("Evaluation Score: {}".format(eval_accuracy))
    print("Test Score: {}".format(test_score))
    if model_version is None and not for_ensemble:
        save_model(eval_accuracy, test_score, conf_rep ,features)
    return eval_accuracy, model, test_score, conf_rep
    
def get_input_output_labels(features):
    with open('data.json', 'r') as f: 
        data = json.load(f)
        x = []
        y = []
        features_arr = []
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

    yaml_info['prediction_model'] = "pretrained_knn_model.pkl"
    yaml_info['features'] = features
    yaml_info['training'] = 'completed'
    yaml_info['name'] = 'knn'
    yaml_info['eval_accuracy'] = float(eval_accuracy)
    yaml_info['test_score'] = float(test_score)
    yaml_info['weight'] = 1
    # yaml_info['conf_rep'] = get_info(conf_rep)

    model_version="knn"
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
        label_data.append({label_information[0]:[float(label_information[1]),float(label_information[2]),float(label_information[3])]})

    return label_data

# train_knn(['nb_of_pixels_per_segment','horizontal_line_intersection','vertical_line_intersection'], for_ensemble=False)