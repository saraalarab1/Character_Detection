import json
import numpy as np 
import pickle
# from skimage import feature
from tqdm import tqdm
from pandas import DataFrame as df
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeavePOut, StratifiedKFold, KFold #for P-cross validation
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix

printable = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
secondLayerLetters = '0OUVWXYZouvwxyz'

def test(X_test, Y_test, model_version):

    if model_version:
        model = pickle.load(open(f'models/{model_version}/pretrained_knn_model.pkl', 'rb' ))
    else:
        model = pickle.load(open('models/knn/pretrained_knn_model.pkl', 'rb' ))

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    for i in range(len(y_pred_proba)):
        print(Y_test[i])
        current_prediction_prob = max(y_pred_proba[i])
        if current_prediction_prob< 0.75:
            possible_results = len([x for x in y_pred_proba[i] if x > 0])
            indices = sort_index(y_pred_proba[i])[:min(possible_results, 3)]
            labels = []
            for index in indices:
                labels.append(printable[index])
            eval_acc, new_model = train_knn(features=['aspect_ratio'], just_train=True, labels= labels)
            y_pred = new_model.predict([X_test[i]])
        print(y_pred_proba[i])
    confusion_matrix(y_pred, Y_test)
    print(confusion_matrix)
    plot_confusion_matrix(model, X_test, Y_test, cmap=plt.cm.Blues)
    plt.show()
    print("Text Prediction: {}".format(y_pred.shape))
    print("Y_test shape: {}".format(Y_test))
    classification_rep = classification_report(Y_test, y_pred,zero_division=True)
    test_score = metrics.accuracy_score(Y_test, y_pred)

    return test_score, classification_rep

def sort_index(lst, rev=True):
    index = range(len(lst))
    s = sorted(index, reverse=rev, key=lambda i: lst[i])
    return s

def train_knn(features, model_version=None, labels = None, just_train = False):
    print('training')
    x,y = get_input_output_labels(features, labels)
    eval_accuracy, model, X_test, Y_test = train(x, y, testing_size=0.25, max_range_k=30, model_version = model_version)
    if not just_train:
        test_score, conf_rep = test(X_test, Y_test, model_version=model_version)
        print("Evaluation Score: {}".format(eval_accuracy))
        print("Test Score: {}".format(test_score))
        print(conf_rep)
    return eval_accuracy, model, 
    
def get_features(x, y, features, labels = None):
    if labels == None: 
        labels = list(printable)
    print('getting features from labels: ' + str(labels))
    with open('data.json', 'r') as f: 
        data = json.load(f)
        features_data = []
        results = []
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
                results.append(y[index])
            index = index + 1
        return features_data, results

def train(features, labels = None):
    with open('data.json', 'r') as f:
        data = json.load(f)
        x = []
        y= []
        for i in data.keys():
            x.append(i)
            y.append(data[i]['label'])
    X0_train, X_test, Y0_train, Y_test = train_test_split(x,y,test_size=0.2, random_state=7)
   
    knn = pickle.load(open(f'models/ensemble/pretrained_ensemble_model.pkl', 'rb' ))
    secondLayer = pickle.load(open(f'models/svm_case/pretrained_svm_model.pkl', 'rb' ))

    X_test_features, Y_test_features = get_features(X_test, Y_test, features)
    X_test_features2, Y_test_features2 = get_features(X_test, Y_test, features)

    Y_pred = knn.predict(X_test_features)
    y_pred_proba = knn.predict_proba(X_test_features)

    cm = confusion_matrix(Y_pred, Y_test)
    # plot_confusion_matrix(knn, X_test_features, Y_test, cmap=plt.cm.Blues)
    plot_confusion_matrix(conf_mat=cm)
    plt.show()

    classification_rep = classification_report(Y_test, Y_pred,zero_division=True)
    score = metrics.accuracy_score(Y_test, Y_pred)
    print("Before Score: ", score)
    # print(classification_rep)

    for i in range(len(y_pred_proba)):
        current_prediction_prob = max(y_pred_proba[i])
        if current_prediction_prob< 0.55:
            if Y_pred[i] in secondLayerLetters:
                possible_results = len([x for x in y_pred_proba[i] if x > 0]) # ['y, Y, q']
                indices = sort_index(y_pred_proba[i])[:min(possible_results, 3)] # y 0.4 q 0.3 Y 0.3                 
                labels = []
                for index in indices:
                    labels.append(printable[index])
                print('possible labels')
                print(labels)
                if labels[0] in secondLayerLetters:
                    new_labels = []
                    for label in labels: 
                        if label in secondLayerLetters:
                            new_labels.append(label)  # y Y
            
                new_prediction = secondLayer.predict([X_test_features2[i]])[0]
                print('new prediction: ' + new_prediction)
                print('old prediction: ' + Y_pred[i])
                print('correct prediction: ' + Y_test[i]) 
                Y_pred[i] = new_prediction
        # print('-----------------------------------------')
        
    classification_rep = classification_report(Y_test, Y_pred,zero_division=True)
    score = metrics.accuracy_score(Y_test, Y_pred)
    print("After Score: ", score)
    # print(classification_rep)
    am = confusion_matrix(Y_pred, Y_test)
    # plot_confusion_matrix(knn, X_test_features, Y_test, cmap=plt.cm.Blues)
    plot_confusion_matrix(conf_mat=am)
    plt.show()


def get_input_output_labels(features, labels = None):
    with open('data.json', 'r') as f: 
        data = json.load(f)
        x = []
        y = []
        for i in data.keys():
            if labels is not None and data[i]['label'] in labels:
                continue
            for feature in features:
                if type(data[i][feature]) != list:
                    data[i][feature] = [data[i][feature]]
                x.append(data[i][feature])
            y.append(data[i]['label'])
    return (x,y)

# train_knn(['nb_of_pixels_per_segment'])

train(features=['nb_of_pixels_per_segment'])