import pickle
from datetime import datetime
from genericpath import exists
from os import makedirs
import os
from flask import Flask, jsonify, render_template, request, redirect, url_for
from matplotlib.style import available
import yaml
import cv2 as cv
from features import get_character_features, get_case_features
from pre_processing import pre_process_image, pre_process_letter
from our_model import predict_model
from knn_classifier import train_knn
from svm_classifier import train_svm
from decision_tree_classifier import train_dt
from flask_cors import CORS
from tensorflow import keras
import base64
from PIL import Image
import io
import numpy as np
from ensemble_classifier import train_ensemble
from ann_model import train_ann
from cnn_model import train_cnn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from segmentation import word_segmentation
from paragraph_segmentation import paragraph_seg
from keras import backend as K
import sklearn
from keras.optimizers import SGD, Adam
from ann_model import create_model

labels=['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']   

app = Flask(__name__)
CORS(app, resources={r"*": {"origins": "*"}})

def read_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        yaml_info = yaml.safe_load(f)
    return yaml_info


@app.route('/info', methods = ['GET'])
def get_available_models():
    """
    This function searches for all models 
    and returns its characterstics
    """
    models = os.listdir('models')
    available_models = dict()
    for model in models:
        if model == "svm_case" or model == "knn_ensemble" or model == "svm_ensemble":
            continue
        if len(os.listdir('models/'+model)) <2:
            continue
        yaml_model_path = os.path.join('models', model, 'model.yaml')
        yaml_info = read_yaml(yaml_model_path)
        if 'features' not in yaml_info:
            yaml_info['features']=[]
        available_models[model] = yaml_info
    response = jsonify(models=models, available_models=available_models)
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response

@app.route('/features', methods = ['GET'])
def get_features():
    """
    This function searches for all models 
    and returns its characterstics
    """
    features_path = os.path.join('features', 'features.yaml')
    features = read_yaml(features_path)
    response = jsonify(features=features['features'])
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response

@app.route('/train_new_model', methods= ['GET','POST'])
def train_new_model():
    if request.method == 'POST':
        models = request.json['models']
        features = request.json['features']
        english = request.json['english']
        arabic = request.json['arabic']
        model_version = str(datetime.now()).replace('-', '_').replace(' ','_').replace(':','_')
        makedirs("models/"+model_version)
        yaml_info = dict()
        prediction_models =[]
        ensemble = False
        if len(models)> 1:
            ensemble = True
            if english:
                prediction_models.append('pretrained_ensemble_english_model.pkl')
            if arabic: 
                prediction_models.append('pretrained_ensemble_arabic_model.pkl')
            yaml_info['prediction_model'] = prediction_models
        else:
            if english:
                prediction_models.append('pretrained_'+models[0]['name']+'_english_model.pkl')
            else:
                prediction_models.append('pretrained_'+models[0]['name']+'_arabic_model.pkl')
            yaml_info['prediction_model'] = prediction_models

        yaml_info['features'] = features
        yaml_info['training'] = 'running'
        yaml_path = os.path.join("models",model_version, 'model.yaml')
        with open(yaml_path, 'w') as output:
            yaml.dump(yaml_info, output)

        estimatorsEnglish=[]
        estimatorsArabic = []
        weights = []
        activation_functions = []
        ensemble_models = []

        for model in models:
            if english:
                if model['name'] == 'knn':
                    eval_accuracy, model_classifier, test_score, conf_rep = train_knn(features, model_version,for_ensemble=ensemble)
                if model['name'] == 'svm':
                    eval_accuracy, model_classifier, test_score, conf_rep = train_svm(features, model_version,for_ensemble=ensemble)
                if model['name'] == 'dt':
                    eval_accuracy, model_classifier, test_score, conf_rep = train_dt(features, model_version,for_ensemble=ensemble)
                if model['name'] == 'ann':
                    activation_functions = model['activation_functions']
                    eval_accuracy, model_classifier, test_score, conf_rep = train_ann(activation_functions,features, model_version,for_ensemble=ensemble)
                if model['name'] == 'cnn':
                    activation_functions = model['activation_functions']
                    eval_accuracy, model_classifier, test_score, conf_rep = train_cnn(activation_functions, model_version,for_ensemble=ensemble)

                estimatorsEnglish.append((model['name'],model_classifier))
                ensemble_models.append(model['name'])
                label_data= get_info(conf_rep)
                weights.append(int(model['weight']))

            if arabic:
                if model['name'] == 'knn':
                    eval_accuracy, model_classifier, test_score, conf_rep = train_knn(features, model_version,for_ensemble=ensemble, arabic = True)
                if model['name'] == 'svm':
                    eval_accuracy, model_classifier, test_score, conf_rep = train_svm(features, model_version,for_ensemble=ensemble, arabic = True)
                if model['name'] == 'dt':
                    eval_accuracy, model_classifier, test_score, conf_rep = train_dt(features, model_version,for_ensemble=ensemble, arabic = True)
                if model['name'] == 'ann':
                    activation_functions = model['activation_functions']
                    eval_accuracy, model_classifier, test_score, conf_rep = train_ann(activation_functions,features, model_version,for_ensemble=ensemble, arabic = True)
                if model['name'] == 'cnn':
                    activation_functions = model['activation_functions']
                    eval_accuracy, model_classifier, test_score, conf_rep = train_cnn(activation_functions, model_version,for_ensemble=ensemble, arabic = True)

                estimatorsArabic.append((model['name'],model_classifier)) 
                ensemble_models.append(model['name'])
                label_data= get_info(conf_rep)
                if not english:
                    weights.append(int(model['weight']))

            if not ensemble:
                yaml_info['name'] = model['name']
                yaml_info['eval_accuracy'] = float(eval_accuracy)
                yaml_info['test_score'] = float(test_score)
                yaml_info['weight'] = model['weight']
                yaml_info['conf_rep'] = label_data
            
        if ensemble:
            prediction_models = yaml_info['prediction_model']
            print(prediction_models)
            for prediction_model in prediction_models:
                if prediction_model == 'pretrained_ensemble_english_model.pkl':
                    eval_accuracy, test_score, conf_rep = train_ensemble(estimatorsEnglish,weights,features,model_version)
                if prediction_model == 'pretrained_ensemble_arabic_model.pkl':
                    eval_accuracy, test_score, conf_rep = train_ensemble(estimatorsArabic, weights, features, model_version, arabic= True)
            yaml_info['name'] = 'ensemble'
            yaml_info['eval_accuracy'] = float(eval_accuracy)
            yaml_info['test_score'] = float(test_score)
            yaml_info['weights'] = weights
            yaml_info['ensemble_models'] = ensemble_models
            yaml_info['conf_rep'] = label_data

        yaml_info['training'] = 'completed'
        with open(yaml_path, 'w') as output:
            yaml.dump(yaml_info, output)

    response = jsonify(training='completed', eval_accuracy=eval_accuracy, test_score=test_score)
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response
    
@app.route('/predict', methods=['GET','POST'])
def predict():
    print('predicting')
    if request.method == 'POST':
        base64_image = request.json['image']
        base64_image = base64_image.split('base64,')[1]
        im_bytes = base64.b64decode(base64_image)
        im_arr = np.frombuffer(im_bytes, dtype=np.uint8)  # im_arr is one-dim Numpy array
        image = cv.imdecode(im_arr, flags=cv.IMREAD_COLOR)
        model_version = request.json['model_version']
        category = request.json['category']
        print(category)
        if category == "letter":
            letters = [[image]]
        elif category == "word": 
            letters = [word_segmentation(image)]
        else:
            letters = paragraph_seg(image)

        for i in range(len(letters)):
            for j in range(len(letters[i])):
                letters[i][j] = pre_process_letter(letters[i][j])

        print(model_version)
        yaml_path = os.path.join(f"models/{model_version}", "model.yaml")
        with open(yaml_path, 'r') as f:
            yaml_info = yaml.safe_load(f)
            print(model_version, yaml_info['prediction_model'])
            features = yaml_info['features']
            character_features = get_character_features(features, letters)
            case_features = get_case_features(letters)
            if model_version != 'topPerformer':
                model_names = yaml_info['prediction_model']
            output = ''
            if model_version == 'topPerformer':
                for i in range(len(character_features)):
                    prediction = predict_model(character_features[i], case_features[i])
                    output = output + prediction[0]
            else:
                probability = 0
                for model_name in model_names:
                    current_words = ''
                    current_prediction = ''
                    current_probability = 0
                    model = pickle.load(open(os.path.join(f"models/{model_version}", model_name), 'rb' ))
                    if 'cnn' in model_name:
                        for w in letters:
                            for letter in w:
                                letter = np.array([letter])
                                current_prediction = model.predict(letter)
                                print(current_prediction)
                                current_prediction = labels[current_prediction[0]]
                                current_probability = current_probability + max(model.predict_proba(letter)[0])
                                if model_name.__contains__('arabic'):
                                    current_words = current_prediction[0] + current_words
                                else:
                                    current_words = current_words + current_prediction[0]
                    else:
                        for i in range(len(character_features)):
                            for j in range(len(character_features[i])):
                                scaler_path = os.path.join(f"models/{model_version}/scaler.pkl")
                                character_feature = character_features[i][j]
                                if os.path.exists(scaler_path):
                                    scaling = pickle.load(open(scaler_path, 'rb'))
                                    character_feature = scaling.transform(character_features[i][j])
                                current_prediction = model.predict(character_feature)
                                if 'ann' in model_name:
                                    current_prediction = labels[current_prediction[0]]
                                current_probability = current_probability + max(model.predict_proba(character_feature)[0])
                                if model_name.__contains__('arabic'):
                                    current_words = current_prediction[0] + current_words
                                else:
                                    current_words = current_words + current_prediction[0]
                    current_words = current_words + " "
                    print(current_words)
                    if current_probability > probability:
                        probability = current_probability
                        output = current_words
                    current_prediction = ''
                    current_probability = 0
    response = jsonify(word = output)
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


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
            break
        label_data.append({label_information[0]:[float(label_information[1]),float(label_information[2]),float(label_information[3])]})

    return label_data


# Take in base64 string and return PIL image
def stringToImage(base64_string):
    imgdata = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(imgdata))

# convert PIL Image to an RGB image( technically a numpy array ) that's compatible with opencv
def toRGB(image):
    return cv.cvtColor(np.array(image), cv.COLOR_BGR2RGB)

if __name__ == "__main__":
    app.run(debug = True)