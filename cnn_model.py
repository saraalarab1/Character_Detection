from keras.datasets import mnist
import matplotlib.pyplot as plt
import cv2
import numpy as np
import json
import yaml
import os
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from keras.optimizers import SGD, Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.utils import to_categorical, np_utils
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler


def train(X, Y, testing_size, for_ensemble,model_version):


    # Reshaping the data in csv file so that it can be displayed as an image...

    X0_train, X_test, Y0_train, Y_test = train_test_split(X, Y, test_size = testing_size)
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


    # CNN model...

    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)))
    model.add(MaxPool2D(pool_size=(2, 2), strides=2))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=2))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding = 'valid'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=2))

    model.add(Flatten())

    model.add(Dense(64,activation ="relu"))
    model.add(Dense(128,activation ="relu"))

    model.add(Dense(26,activation ="softmax"))



    model.compile(optimizer = Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.0001)
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')


    history = model.fit(X0_train, Y0_train, epochs=1, callbacks=[reduce_lr, early_stop],  validation_data = (X_test,Y_test))


    model.summary()
    model.save(r'model_hand.h5')

    print("The validation accuracy is :", history.history['val_accuracy'])
    print("The training accuracy is :", history.history['accuracy'])
    print("The validation loss is :", history.history['val_loss'])
    print("The training loss is :", history.history['loss'])

    test_loss, test_accuracy = model.evaluate(X_test, Y_test)

    #save the pretrained model:
    model_name='pretrained_cnn_model.pkl'
    if model_version:
        pickle.dump(model, open(f"models/{model_version}/{model_name}", 'wb'))
    elif for_ensemble:
        pickle.dump(model, open(f"models/cnn_ensemble/{model_name}", 'wb'))
    else:
        pickle.dump(model, open(f"models/cnn/{model_name}", 'wb'))

    return test_accuracy, model, X_test, Y_test

def test(X_test, Y_test, model_version, for_ensemble):

    if model_version: 
        model = pickle.load(open(f'models/{model_version}/pretrained_cnn_model.pkl', 'rb' ))
    elif for_ensemble:
        model = pickle.load(open('models/cnn_ensemble/pretrained_cnn_model.pkl', 'rb' ))
    else:
        model = pickle.load(open('models/cnn/pretrained_cnn_model.pkl', 'rb' ))

    y_pred = model.predict(X_test)
    classification_rep = 0
    test_score = 0
    # classification_rep = classification_report(Y_test, y_pred,zero_division=True)
    # test_score = metrics.accuracy_score(Y_test, y_pred)

    return test_score, classification_rep

def save_model(eval_accuracy, test_score, conf_rep, for_ensemble, features ):
    yaml_info = dict()

    yaml_info['prediction_model'] = "pretrained_ensemble_model.pkl"
    yaml_info['features'] = features
    yaml_info['training'] = 'completed'
    yaml_info['name'] = 'cnn'

    model_version="cnn"
    if for_ensemble:
        model_version = "cnn_ensemble"

    yaml_path = os.path.join("models",model_version, 'model.yaml')
    with open(yaml_path, 'w') as output:
        yaml.dump(yaml_info, output)

        yaml_info['cnn'] = dict()
        yaml_info['cnn']['eval_accuracy'] = float(eval_accuracy)
        yaml_info['cnn']['test_score'] = float(test_score)
        # yaml_info['cnn']['conf_rep'] = get_info(conf_rep)
        yaml_info['cnn']['weight'] = 1

    with open(yaml_path, 'w') as output:
        yaml.dump(yaml_info, output)

## test accuracy
def train_cnn(features, model_version=None, for_ensemble = False):
    print('training')
    x,y = get_input_output_labels(features)
    y_enc = prepare_targets(y)
    eval_accuracy, model, X_test, Y_test = train(x, y_enc, testing_size=0.2,model_version = model_version, for_ensemble=for_ensemble)
    # test_score, conf_rep = test(X_test, Y_test, model_version=model_version,for_ensemble = for_ensemble)
    # print(conf_rep)
    # print("Evaluation Score: {}".format(eval_accuracy))
    # print("Test Score: {}".format(test_score))
    # if model_version is None:
    #     save_model(eval_accuracy, test_score, conf_rep,for_ensemble ,features)
    # return eval_accuracy, model, test_score, conf_rep

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


train_cnn(['nb_of_pixels_per_segment','horizontal_line_intersection','vertical_line_intersection'])