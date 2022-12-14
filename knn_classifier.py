
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

secondLayerLetters = 'uUvVxXwWyYzZ0oO'

def train(x, y, testing_size, model_version, optimal_k=True, max_range_k=100):
    
    X0_train, X_test, Y0_train, Y_test = train_test_split(x,y,test_size=testing_size, random_state=7)
    #Scaler is needed to scale all the inputs to a similar range
    # scaler = StandardScaler()
    # scaler = scaler.fit(X0_train)
    # X0_train = scaler.transform(X0_train)
    # X_test = scaler.transform(X_test)
    #X_train, X_eval, Y_train, y_eval = train_test_split(X0_train, Y0_train, test_size= 100/k_cross_validation_ratio, random_state=7)

    # range for optimal K, can be specified by user
    if optimal_k and max_range_k>1:
        k_range= range(1, max_range_k)
    else:
        k_range=range(1,50)
    
    scores = {}
    scores_list = []

    # Set the parameters by cross-validation

    #finding the optimal nb of neighbors
    for k in tqdm(k_range):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X0_train, Y0_train)
        y_pred = knn.predict(X_test)
        scores[k] = metrics.accuracy_score(Y_test, y_pred)
        scores_list.append(round(metrics.accuracy_score(Y_test, y_pred),3))

    k_optimal = scores_list.index(max(scores_list)) +1
    print(f"K Optimal: {k_optimal}")
    model = KNeighborsClassifier(n_neighbors= k_optimal)
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
    skf = StratifiedKFold(n_splits=10, random_state=None)
    skf.get_n_splits(X0_train, Y0_train)
    for train_index, test_index in skf.split(X0_train,Y0_train):
    
        # print("TRAIN:", train_index, "Validation:", test_index)
        X_train, X_eval = df(X0_train).iloc[train_index], df(X0_train).iloc[test_index]
        Y_train, y_eval = df(Y0_train).iloc[train_index], df(Y0_train).iloc[test_index]
    
        model.fit(X_train, Y_train.values.ravel())
        predictions = model.predict(X_eval)
        score = accuracy_score(predictions, y_eval)
        accuracys.append(score)


    eval_accuracy = np.mean(accuracys)

    #save the pretrained model:
    model_name='pretrained_knn_model.pkl'
    if model_version:
        pickle.dump(model, open(f"models/{model_version}/{model_name}", 'wb'))
    else:
        pickle.dump(model, open(f"models/knn/SecondLayerKnn.pkl", 'wb'))

    return eval_accuracy, model, X_test, Y_test


def test(X_test, Y_test, model_version):

    if model_version:
        model = pickle.load(open(f'models/{model_version}/pretrained_knn_model.pkl', 'rb' ))
    else:
        model = pickle.load(open('models/knn/SecondLayerKnn.pkl', 'rb' ))

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    for i in range(len(y_pred)):
        print(f"True Value {Y_test[i]}")
        print(f"Predicted Value {y_pred[i]}")
        print(f"Probability {str(y_pred_proba[i])}")
    print("Text Prediction: {}".format(y_pred.shape))
    print("Y_test shape: {}".format(Y_test))
    classification_rep = classification_report(Y_test, y_pred,zero_division=True)
    test_score = metrics.accuracy_score(Y_test, y_pred)

    return test_score, classification_rep

def train_knn(features, model_version=None):
    print('training')
    x,y = get_input_output_labels(features)
    eval_accuracy, model, X_test, Y_test = train(x, y, testing_size=0.25, max_range_k=100, model_version = model_version)
    test_score, conf_rep = test(X_test, Y_test, model_version=model_version)
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
            if data[i]['label'] in secondLayerLetters:
                for feature in features:
                    arr = data[i][feature]
                    if type(arr) != list:
                        arr = [arr]
                    features_arr.extend(arr)
                x.append(features_arr)
                y.append(data[i]['label'])
    return (x,y)

# train_knn(['nb_of_pixels_per_segment'])
