
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

def train(x, y, testing_size, model_version):

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

    #Evaluation using cross validation
    # LeavePOut
    # lpo = LeavePOut(p=2)
    # KFold
    # kf = KFold(n_splits=10)
    # kf.get_n_splits(X0_train)
    # StratifiedKFold
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

    #save the pretrained model:
    model_name='pretrained_dtree_model.pkl'
    if model_version:
        pickle.dump(model, open(f"models/{model_version}/{model_name}", 'wb'))
    else:
        pickle.dump(model, open(f"models/dt/{model_name}", 'wb'))

    return eval_accuracy, model, X_test, Y_test


def test(X_test, Y_test, model_version):

    if model_version:
        model = pickle.load(open(f'models/{model_version}/pretrained_dtree_model.pkl', 'rb' ))
    else:
        model = pickle.load(open('models/dt/pretrained_dtree_model.pkl', 'rb' ))

    y_pred = model.predict(X_test)
    print("Text Prediction: {}".format(y_pred.shape))
    print("Y_test shape: {}".format(Y_test))
    classification_rep = classification_report(Y_test, y_pred,zero_division=True)
    test_score = metrics.accuracy_score(Y_test, y_pred)

    return test_score, classification_rep

def train_dt(features, model_version=None):
    print('training')
    x,y = get_input_output_labels(features)
    eval_accuracy, model, X_test, Y_test = train(x, y, testing_size=0.05, model_version = model_version)
    test_score, conf_rep = test(X_test, Y_test, model_version = model_version)
    print("Evaluation Score: {}".format(eval_accuracy))
    print("Test Score: {}".format(test_score))
    print(conf_rep)
    return eval_accuracy, model, test_score, conf_rep


def get_input_output_labels(features):
    with open('data.json', 'r') as f: 
        data = json.load(f)
        x = []
        y = []
        for i in data.keys():
            features_arr = []
            for feature in features:
                arr = data[i][feature]
                features_arr.extend(arr)
            x.append(features_arr)
            y.append(data[i]['label'])
    return (x,y)

train_dt(['nb_of_pixels_per_segment'])
