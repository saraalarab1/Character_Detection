from keras.datasets import mnist
import matplotlib.pyplot as plt
import cv2
import numpy as np
import json
import yaml
import os
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout, Reshape
from keras.optimizers import SGD, Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.utils import to_categorical, np_utils
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler


def train(X, Y,activation_functions,testing_size, for_ensemble,model_version):

    X0_train, X_test, Y0_train, Y_test = train_test_split(X, Y, test_size = testing_size)

  # print("TRAIN:", train_index, "Validation:", test_index)
    # X0_train = pd.DataFrame(X0_train)
    # Y0_train = pd.DataFrame(Y0_train)
    # X_test = pd.DataFrame(X_test)
    # Y_test = pd.DataFrame(Y_test)


    # CNN model...

    model = Sequential()

    # Add the convolutional layers using a for loop
    for i in range(len(activation_functions)):
        # If this is the first convolutional layer, specify the input shape
        if i == 0:
            model.add(Conv2D(filters=32, kernel_size=(3, 3), activation=activation_functions[i], input_shape=X0_train.shape[1:]))
        else:
            # For subsequent convolutional layers, do not specify the input shape
            model.add(Conv2D(filters=64*i, kernel_size=(3, 3), activation=activation_functions[i], padding = 'valid'))

        # Add a max pooling layer
        model.add(MaxPool2D(pool_size=(2, 2),strides=2))

    # Add a flatten layer
    model.add(Flatten())

    # Add the dense layers using a for loop
    for i in range(len(activation_functions)):
        # If this is the first dense layer, specify the number of units
        if i == 0:
            model.add(Dense(128, activation=activation_functions[i]))
        else:
            # For subsequent dense layers, do not specify the number of units
            model.add(Dense(32*i,activation_functions[i]))

    # model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu',  input_shape=X0_train.shape[1:]))
    # model.add(MaxPool2D(pool_size=(2, 2), strides=2))

    # model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same'))
    # model.add(MaxPool2D(pool_size=(2, 2), strides=2))

    # model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding = 'valid'))
    # model.add(MaxPool2D(pool_size=(2, 2), strides=2))

    # model.add(Flatten())

    # model.add(Dense(64,activation ="relu"))
    # model.add(Dense(128,activation ="relu"))

    # model.add(Dense(26,activation ="softmax"))

    model.compile(optimizer = Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.0001)
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')


    history = model.fit(X0_train, Y0_train, epochs=1, callbacks=[reduce_lr, early_stop])

    # model.summary()

    eval_accuracy = np.mean(history.history['accuracy'])

    # print("The validation loss is :", history.history['val_loss'])
    # print("The training loss is :", history.history['loss'])

    #save the pretrained model:
    model_name=r'pretrained_cnn_model.h5'
    if model_version:
        model.save(f"models/{model_version}/{model_name}")
    elif for_ensemble:
        model.save(f"models/cnn_ensemble/{model_name}")
    else:
        model.save(f"models/cnn/{model_name}")

    return eval_accuracy, model, X_test, Y_test

def test(X_test, Y_test, model_version, for_ensemble):

    if model_version: 
        model = keras.models.load_model(f'models/{model_version}/pretrained_cnn_model.h5')
    elif for_ensemble:
        model = keras.models.load_model(f'models/ann_ensemble/pretrained_cnn_model.h5')    
    else:
        model = keras.models.load_model(f'models/ann/pretrained_cnn_model.h5')

    score = model.evaluate(X_test, Y_test, batch_size=8)
    test_score = score[1]
    test_loss = score[0]
    classification_rep = 0
    # classification_rep = classification_report(Y_test, y_pred,zero_division=True)

    return test_score, classification_rep

def save_model(eval_accuracy, test_score, conf_rep, features ):
    yaml_info = dict()

    yaml_info['prediction_model'] = "pretrained_ensemble_model.h5"
    yaml_info['features'] = features
    yaml_info['training'] = 'completed'
    yaml_info['name'] = 'cnn'
    yaml_info['eval_accuracy'] = float(eval_accuracy)
    yaml_info['test_score'] = float(test_score)
    yaml_info['weight'] = 1
    # yaml_info['conf_rep'] = get_info(conf_rep)

    model_version="cnn"
    yaml_path = os.path.join("models",model_version, 'model.yaml')
    with open(yaml_path, 'w') as output:
        yaml.dump(yaml_info, output)

## test accuracy
def train_cnn(activation_functions,features, model_version=None, for_ensemble = False):
    print('training')
    x,y = get_input_output_labels(features)
    y_enc = prepare_targets(y)
    print(len(x))
    print(len(y))
    model, eval_accuracy, test_score,  = train(x, y_enc,activation_functions,testing_size=0.2,model_version = model_version, for_ensemble=for_ensemble)
    conf_rep = False
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
    x = np.load('cnn_data.npy')
    with open('data.json', 'r') as f: 
        data = json.load(f)
        y = []
        for i in data.keys():
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


# train_cnn(['relu','relu','relu'],['nb_of_pixels_per_segment','horizontal_line_intersection','vertical_line_intersection'])