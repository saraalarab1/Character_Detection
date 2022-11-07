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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


printable = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
SimilarLowerUpperCase = 'uUvVzZxXkKjJnNmM0OoPpSsCcYy'

def test(X_test,Y_test, features):

    ensemble = pickle.load(open(f'models/ensemble/pretrained_ensemble_model.pkl', 'rb' ))
    svm_case = pickle.load(open(f'models/svm_case/pretrained_svm_model.pkl', 'rb' ))

    X_test_features = get_features(X_test, Y_test, features)

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
                new_prediction = svm_case.predict([X_test_features[i]])[0]
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
    print(classification_rep)
    score = metrics.accuracy_score(Y_test, Y_pred)
    print("After Score: ", score)

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


def sort_index(lst, rev=True):
    index = range(len(lst))
    s = sorted(index, reverse=rev, key=lambda i: lst[i])
    return s

def test_model(features):
    print('training')
    x,y = get_input_output_labels()
    X0_train, X_test, Y0_train, Y_test = train_test_split(x,y,test_size=0.8, random_state=7)
    test(X_test,Y_test, features)

    
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


test_model(features=['nb_of_pixels_per_segment'])