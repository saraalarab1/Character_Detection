import json
import numpy as np 
import pickle
from pandas import DataFrame as df
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import  StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
import yaml


printable = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
SimilarLowerUpperCase = 'uUvVzZxXkKjJnNmM0OoPpSsCcYy'

def test(X_test,Y_test, features):

    ensemble = pickle.load(open(f'models/topPerformer/pretrained_ensemble_model.pkl', 'rb' ))
    case = pickle.load(open(f'models/topPerformer/pretrained_svm_model.pkl', 'rb' ))

    X_test_features = get_features(X_test, Y_test, features)
    X_test_features_case = get_features(X_test, Y_test, features=['nb_of_pixels_per_segment','aspect_ratio'])


    Y_pred = ensemble.predict(X_test_features)
    y_pred_proba = ensemble.predict_proba(X_test_features)
    
    for i in range(len(y_pred_proba)):
        current_prediction_prob = max(y_pred_proba[i])
        # check if probability of predicted y is less than 60%
        if current_prediction_prob< 0.6:
            # check if prediction belongs to similar lower/upper case characters
            if Y_pred[i] in SimilarLowerUpperCase:
                nb_of_possible_results = len([x for x in y_pred_proba[i] if x > 0]) # ['y, Y, q']
                # get index of the highest three (or less) probabilities
                indices = sort_index(y_pred_proba[i])[:min(nb_of_possible_results, 3)] # y 0.4 q 0.3 Y 0.3                 
                labels = []
                # get labels from index
                for index in indices:
                    labels.append(printable[index])
                new_labels = []
                # if highest probability is in upper/lower case, we want to check again
                if labels[0] in SimilarLowerUpperCase:
                    for label in labels: 
                        if label in SimilarLowerUpperCase:
                            new_labels.append(label)  # y Y
                # if we only got one label belonging to similar upper/lower case characters, we don't need to proceed
                if len(new_labels)<=1:
                    continue
                
                # classify highest probability prediction to upper or lower based on svm model
                new_prediction = case.predict([X_test_features_case[i]])[0]
                if new_prediction == 'lower':
                    print('new prediction: ' + Y_pred[i].lower())
                else:
                    print('new prediction: ' + Y_pred[i].upper())
                print('old prediction: ' + Y_pred[i])
                print('correct prediction: ' + Y_test[i]) 
                Y_pred[i] = Y_pred[i].lower() if new_prediction == 'lower' else Y_pred[i].upper()
                # Y_pred[i] = Y_test[i]
                print('-----------------------------------------')

    classification_rep = classification_report(Y_test, Y_pred,zero_division=True)
    test_score = metrics.accuracy_score(Y_test, Y_pred)
    print(classification_rep)
    print("After Score: ", test_score)



    cm = confusion_matrix(Y_pred, Y_test)
    labels=['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']   
    cmd = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labels)
    cmd.plot()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    return classification_rep, test_score


def predict_model(features, features_case, category):
    ensemble = pickle.load(open(f'models/topPerformer/pretrained_ensemble_model.pkl', 'rb' ))
    case = pickle.load(open(f'models/topPerformer/pretrained_svm_model.pkl', 'rb' ))
    language = pickle.load(open('models/language_model/pretrained_language_model.pkl', 'rb'))
    arabic_ensemble = pickle.load(open(f'models/topPerformer/pretrained_ensemble_arabic_model.pkl', 'rb' ))
    scaling = pickle.load(open('models/language_model/scaler.pkl', 'rb'))
    Y_pred = ensemble.predict(features)
    y_pred_proba = ensemble.predict_proba(features)
    features = scaling.transform(features)
    Y_pred_language = language.predict(features)
    Y_pred_arabic = arabic_ensemble.predict(features)
    if Y_pred_language == 'english' or category == 'paragraph':
        for i in range(len(y_pred_proba)):
            current_prediction_prob = max(y_pred_proba[i])
            # check if probability of predicted y is less than 60%
            if current_prediction_prob< 0.6:
                # check if prediction belongs to similar lower/upper case characters
                if Y_pred[i] in SimilarLowerUpperCase:
                    nb_of_possible_results = len([x for x in y_pred_proba[i] if x > 0]) # ['y, Y, q']
                    # get index of the highest three (or less) probabilities
                    indices = sort_index(y_pred_proba[i])[:min(nb_of_possible_results, 3)] # y 0.4 q 0.3 Y 0.3                 
                    labels = []
                    # get labels from index
                    for index in indices:
                        labels.append(printable[index])
                    new_labels = []
                    # if highest probability is in upper/lower case, we want to check again
                    if labels[0] in SimilarLowerUpperCase:
                        for label in labels: 
                            if label in SimilarLowerUpperCase:
                                new_labels.append(label)  # y Y
                    # if we only got one label belonging to similar upper/lower case characters, we don't need to proceed
                    if len(new_labels)<=1:
                        continue
                    
                    # classify highest probability prediction to upper or lower based on svm model
                    print('feature case is: '+ str(features_case[i]))
                    new_prediction = case.predict([[features_case[i]]])[0]
                    if new_prediction == 'lower':
                        print('new prediction: ' + Y_pred[i].lower())
                    else:
                        print('new prediction: ' + Y_pred[i].upper())
                    print('old prediction: ' + Y_pred[i])
                    Y_pred[i] = Y_pred[i].lower() if new_prediction == 'lower' else Y_pred[i].upper()
                    # Y_pred[i] = Y_test[i]
    else:
        return Y_pred_arabic
    return Y_pred

def sort_index(lst, rev=True):
    index = range(len(lst))
    s = sorted(index, reverse=rev, key=lambda i: lst[i])
    return s

def test_model(features):
    print('training')
    x,y = get_input_output_labels()
    X0_train, X_test, Y0_train, Y_test = train_test_split(x,y,test_size=0.2, random_state=7)
    conf_rep, test_score = test(X_test,Y_test, features)
    save_model(test_score, conf_rep ,features)
    
def get_features(x, y, features, labels = None):
    if labels == None: 
        labels = list(printable)
    print('getting features from labels: ' + str(labels))
    with open('data.json', 'r') as f: 
        data = json.load(f)
        features_data = []
        index = 0
        for i in x:
            features_list = []
            if data[i]['label'] in labels:
                features_list = []
                for feature in features:
                    if type(data[i][feature]) != list:
                        features_list.append(data[i][feature])
                    else:
                        features_list = data[i][feature]

                features_data.append(features_list)
            index = index + 1
        return features_data

def get_input_output_labels():
    with open('data.json', 'r') as f:
        data = json.load(f)
        x = []
        y= []
        for i in data.keys():
            x.append(i)
            y.append(data[i]['label'])
    return (x,y)

def save_model(test_score, conf_rep, features):
    yaml_info = dict()

    yaml_info['prediction_model'] = ["pretrained_detection_model.pkl"]
    yaml_info['features'] = features
    yaml_info['training'] = 'completed'
    yaml_info['name'] = 'our_model'
    yaml_info['test_score'] = float(test_score)
    yaml_info['weight'] = 1
    yaml_info['conf_rep'] = get_info(conf_rep)

    model_version="our_model"

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


# test_model(features=['nb_of_pixels_per_segment','horizontal_line_intersection','vertical_line_intersection'])