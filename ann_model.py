import json
import yaml
import numpy as np 
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import os
# from skimage import feature
import pickle
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import classification_report


def train(X, Y,activation_functions, testing_size, for_ensemble,model_version):
    
    X0_train, X_test, Y0_train, Y_test = train_test_split(X,Y,test_size=testing_size, random_state=7)
    #Scaler is needed to scale all the inputs to a similar range
    scaler = StandardScaler()
    scaler = scaler.fit(X0_train)
    X0_train = scaler.transform(X0_train)
    X_test = scaler.transform(X_test)

  # print("TRAIN:", train_index, "Validation:", test_index)
    X0_train = pd.DataFrame(X0_train)
    Y0_train = pd.DataFrame(Y0_train)
    X_test = pd.DataFrame(X_test)
    Y_test = pd.DataFrame(Y_test)

    # Build the ANN model
    model = Sequential()
    # Add the input layer
    model.add(Dense(100, input_dim=X0_train.shape[1], activation=activation_functions[0]))

    # Add the hidden layers
    for i in range(2, len(activation_functions)):
        model.add(Dense(88-i*3, activation=activation_functions[i]))

    # Add the output layer
    model.add(Dense(62, activation=activation_functions[1]))

    # Compile the model
    model.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

    # Train the model on the training set
    history =  model.fit(X0_train, Y0_train, epochs=50, batch_size=8)

    eval_accuracy = np.mean(history.history['accuracy'])

    # model.summary()

    #save the pretrained model:
    model_name=r'pretrained_ann_model.h5'
    if model_version:
        model.save(f"models/{model_version}/{model_name}")
    elif for_ensemble:
        model.save(f"models/ann_ensemble/{model_name}")
    else:
        model.save(f"models/ann/{model_name}")

    return eval_accuracy, model, X_test, Y_test


def test(X_test, Y_test, model_version, for_ensemble):

    if model_version: 
        model = keras.models.load_model(f'models/{model_version}/pretrained_ann_model.h5')
    elif for_ensemble:
        model = keras.models.load_model(f'models/ann_ensemble/pretrained_ann_model.h5')    
    else:
        model = keras.models.load_model(f'models/ann/pretrained_ann_model.h5')

    score = model.evaluate(X_test, Y_test, batch_size=8)
    test_score = score[1]
    test_loss = score[0]
    classification_rep = 0
    # classification_rep = classification_report(Y_test, y_pred,zero_division=True)

    return test_score, classification_rep


def save_model(eval_accuracy, test_score, conf_rep, features):
    yaml_info = dict()

    yaml_info['prediction_model'] = ["pretrained_ann_model.h5"]
    yaml_info['features'] = features
    yaml_info['training'] = 'completed'
    yaml_info['name'] = 'ann'
    yaml_info['eval_accuracy'] = float(eval_accuracy)
    yaml_info['test_score'] = float(test_score)
    yaml_info['weight'] = 1
    # yaml_info['conf_rep'] = get_info(conf_rep)

    model_version="ann"
    yaml_path = os.path.join("models",model_version, 'model.yaml')
    with open(yaml_path, 'w') as output:
        yaml.dump(yaml_info, output)

## test accuracy
def train_ann(activation_functions,features, model_version=None, for_ensemble = False, arabic = False):
    print('training')
    x,y = get_input_output_labels(features)
    y_enc = prepare_targets(y)
    eval_accuracy, model, X_test, Y_test = train(x, y_enc,activation_functions, testing_size=0.2,model_version = model_version, for_ensemble=for_ensemble)
    test_score, conf_rep = test(X_test, Y_test, model_version=model_version,for_ensemble = for_ensemble)
    print(conf_rep)
    print("Evaluation Score: {}".format(eval_accuracy))
    print("Test Score: {}".format(test_score))
    if model_version is None and not for_ensemble:
        save_model(eval_accuracy, test_score, conf_rep ,features)
    return eval_accuracy, model, test_score, conf_rep

def prepare_targets(y):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    return y_enc

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

def get_info(conf_rep):
    data = conf_rep.splitlines()[2:61]
    label_data = []
    for label_information in data:
        label_information = " ".join(label_information.split())
        label_information = label_information.split(" ")
        if len(label_information) < 4:
            continue
        label_data.append({label_information[0]:[float(label_information[1]),float(label_information[2]),float(label_information[3])]})

    return label_data


# train_ann(['nb_of_pixels_per_segment','horizontal_line_intersection','vertical_line_intersection'])