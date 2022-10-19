
import json
import numpy as np 
import cv2
import pandas as pd
import pickle
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeavePOut #for P-cross validation
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

def get_gamma_and_C(model):

    # creating a KFold object with 5 splits 
    folds = KFold(n_splits = 5, shuffle = True, random_state = 101)

    #specify range of hyperparameters
    hyper_params = [ {'gamma': [1e-2, 1e-3, 1e-4],
                        'C': [1, 10, 100, 1000]}]

    model = GridSearchCV(
        estimator= model, 
        param_grid= hyper_params,
        scoring= 'accuracy',
        cv = folds, 
        verbose= 1,
        return_train_score= True)

    return model.best_params_



def train(X, Y, k_cross_validation_ratio, testing_size):

    X0_train, X_test, Y0_train, Y_test = train_test_split(X,Y,test_size=testing_size, random_state=7)
    #Scaler is needed to scale all the inputs to a similar range
    # scaler = StandardScaler()
    # scaler = scaler.fit(X0_train)
    # X0_train = scaler.transform(X0_train)
    # X_test = scaler.transform(X_test)
    #X_train, X_eval, Y_train, y_eval = train_test_split(X0_train, Y0_train, test_size= 100/k_cross_validation_ratio, random_state=7)
    
    model = SVC(kernel = 'rbf', random_state = 0)

    #eval_score_list = []
    #Evaluation using cross validation: lpo: leave p out
    lpo = LeavePOut(p=1)
    accuracys=[]

    skf = StratifiedKFold(n_splits=10, random_state=None)
    skf.get_n_splits(X0_train, Y0_train)
    for train_index, test_index in skf.split(X0_train, Y0_train):
    
        # print("TRAIN:", train_index, "Validation:", test_index)
        X_train, X_eval = pd.DataFrame(X0_train).iloc[train_index], pd.DataFrame(X0_train).iloc[test_index]
        Y_train, y_eval = pd.DataFrame(Y0_train).iloc[train_index], pd.DataFrame(Y0_train).iloc[test_index]
    
        model.fit(X0_train, Y0_train)
        predictions = model.predict(X_eval)
        score = accuracy_score(predictions, y_eval)
        print(score)
        accuracys.append(score)
        #scores = cross_val_score(knn, X, Y, cv=5, scoring='accuracy')
        #eval_score_list.append(scores.mean())

    #eval_accuracy = np.mean(eval_score_list)
    eval_accuracy = np.mean(accuracys)

    #save the pretrained model:
    model_name='pretrained_svm_model'
    pickle.dump(model, open(model_name, 'wb'))

    return eval_accuracy, model, X0_train, Y0_train, X_test, Y_test


def test(X_train, Y_train, X_test, Y_test,pretrain_model=False):

    if pretrain_model:
        model = pickle.load(open('pretrained_svm_model', 'rb' ))
        
    else:
        eval_score, model, X_train, Y_train, X_test, Y_test = train(X_test, Y_test, pretrained_model=False)
        print("Evaluation score: {}".format(eval_score))

    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    print("Text Prediction: {}".format(y_pred.shape))
    print("Y_test shape: {}".format(Y_test))
    classification_rep = classification_report(Y_test, y_pred)
    test_score = metrics.accuracy_score(Y_test, y_pred)

    return test_score, classification_rep


with open('data.json', 'r') as f: 
    data = json.load(f)
    x = []
    y = []
    for i in data.keys():
        arr_2 = data[i]['vertical_histogram_projection']
        arr_3 = data[i]['horizontal_histogram_projection']
        arr_4 = data[i]['horizontal_ratio']
        arr_5 = data[i]['vertical_ratio']
        arr_6 = data[i]['vertical_symmetry']
        arr_7 = data[i]['horizontal_symmetry']

        x.append(data[i]["nb_of_pixels_per_segment"])
        y.append(data[i]['label'])

eval_accuracy, model, X_train, Y_train, X_test, Y_test = train(x, y, k_cross_validation_ratio=5, testing_size=0.2)
test_score, conf_rep = test(X_train, Y_train, X_test, Y_test, pretrain_model=True)
print("Evaluation Score: {}".format(eval_accuracy))
print("Test Score: {}".format(test_score))
print(conf_rep)
