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
from sklearn import metrics
from sklearn.metrics import classification_report
from keras.wrappers.scikit_learn import KerasClassifier

activation_functions = []
shape = None

def create_model():
    global activation_functions
    global shape
  # CNN model
    model = Sequential()
    # first convolutional layer, specify the input shape
    model.add(Conv2D(filters=50, kernel_size=(2, 2), activation=activation_functions[0], input_shape=shape))
    model.add(MaxPool2D(pool_size=(2, 2)))

    # Add the convolutional layers using a for loop
    for i in range(2,len(activation_functions)):
        # For subsequent convolutional layers, do not specify the input shape
        model.add(Conv2D(filters=62*i, kernel_size=(2, 2), activation=activation_functions[i]))
        # Add a max pooling layer
        model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=138, kernel_size=(2, 2), activation=activation_functions[1]))
    model.add(MaxPool2D(pool_size=(2, 2)))

    # Add a flatten layer
    model.add(Flatten())

    # first dense layer, specify the number of units
    model.add(Dense(100, activation=activation_functions[0]))

    # Add the dense layers using a for loop
    for i in range(2,len(activation_functions)):
        # For subsequent dense layers, do not specify the number of units
        model.add(Dense(88-i*5,activation_functions[i]))
    model.add(Dense(62,activation_functions[1]))

    model.compile(optimizer = Adam(learning_rate=0.001), loss=keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

    return model

def train(X, Y,activation_functions_model,testing_size, for_ensemble,model_version,arabic):
    global activation_functions
    global shape
    X0_train, X_test, Y0_train, Y_test = train_test_split(X, Y, test_size = testing_size)

    activation_functions = activation_functions_model
    shape = X0_train.shape[1:]
    classifier = KerasClassifier(build_fn=create_model, epochs=10)
    classifier._estimator_type = "classifier"
    history = classifier.fit(X0_train, Y0_train)

    eval_accuracy = np.mean(history.history['accuracy'])

    # Access the loss values
    loss = history.history['loss']

    # Plot the loss over time
    plt.plot(loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    model_language = 'english'
    if arabic: 
        model_language = 'arabic'
    #save the pretrained model:
    model_name=f'pretrained_cnn_{model_language}_model.pkl'
    if model_version:
        pickle.dump(classifier, open(f"models/{model_version}/{model_name}", 'wb'))
    elif for_ensemble:
        pickle.dump(classifier, open(f"models/cnn_ensemble/{model_name}", 'wb'))
    else:
        pickle.dump(classifier, open(f"models/cnn/{model_name}", 'wb'))

    return eval_accuracy, classifier, X_test, Y_test

def test(X_test, Y_test, model_version, for_ensemble,arabic):
    model_language = 'english'
    if arabic: 
        model_language = 'arabic'
    if model_version: 
        model = pickle.load(open(f'models/{model_version}/pretrained_cnn_{model_language}_model.pkl', 'rb' ))
    elif for_ensemble:
        model = pickle.load(open(f'models/cnn_ensemble/pretrained_cnn_{model_language}_model.pkl', 'rb' ))
    else:
        model = pickle.load(open(f'models/cnn/pretrained_cnn_{model_language}_model.pkl', 'rb' ))

    model.fit(X_test, Y_test)
    y_pred = model.predict(X_test)
    classification_rep = classification_report(Y_test, y_pred,zero_division=True)
    test_score = metrics.accuracy_score(Y_test, y_pred)

    return test_score, classification_rep

def save_model(eval_accuracy, test_score, conf_rep, features,arabic ):
    yaml_info = dict()
    model_language = 'english'
    if arabic: 
        model_language = 'arabic'
    yaml_info['prediction_model'] = [f"pretrained_cnn_{model_language}_model.pkl"]
    yaml_info['features'] = features
    yaml_info['training'] = 'completed'
    yaml_info['name'] = 'cnn'
    yaml_info['eval_accuracy'] = float(eval_accuracy)
    yaml_info['test_score'] = float(test_score)
    yaml_info['weight'] = 1
    yaml_info['conf_rep'] = get_info(conf_rep)

    model_version="cnn"
    yaml_path = os.path.join("models",model_version, 'model.yaml')
    with open(yaml_path, 'w') as output:
        yaml.dump(yaml_info, output)

## test accuracy
def train_cnn(activation_functions,features, model_version=None, for_ensemble = False, arabic = False):
    print('training')
    x,y = get_input_output_labels(arabic)
    y_enc = prepare_targets(y)
    eval_accuracy, model, X_test, Y_test = train(x, y_enc,activation_functions,testing_size=0.2,model_version = model_version, for_ensemble=for_ensemble,arabic=arabic)
    test_score, conf_rep = test(X_test, Y_test, model_version=model_version,for_ensemble = for_ensemble,arabic=arabic)
    print(conf_rep)
    print("Evaluation Score: {}".format(eval_accuracy))
    print("Test Score: {}".format(test_score))
    if model_version is None and not for_ensemble:
        save_model(eval_accuracy, test_score, conf_rep ,features,arabic)
    return eval_accuracy, model, test_score, conf_rep

def prepare_targets(y):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    return y_enc

def get_input_output_labels(arabic):
    data_file = 'data.json'
    if arabic: 
        data_file = 'arabic_data.json'
    x = np.load('cnn_data.npy')
    with open(data_file, 'r') as f: 
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


# train_cnn(['relu','sigmoid','sigmoid'],['nb_of_pixels_per_segment','horizontal_line_intersection','vertical_line_intersection'])