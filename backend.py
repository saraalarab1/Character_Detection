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
from pre_processing import pre_process_image
from our_model import predict_model
from knn_classifier import train_knn
from svm_classifier import train_svm
from decision_tree_classifier import train_dt
from flask_cors import CORS
import base64
from PIL import Image
import io
import numpy as np
from ensemble_classifier import train_ensemble
from ann_model import train_ann
from cnn_model import train_cnn

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
        model_version = str(datetime.now()).replace('-', '_').replace(' ','_').replace(':','_')
        makedirs("models/"+model_version)
        yaml_info = dict()
        if len(models)> 1:
            yaml_info['prediction_model'] = 'pretrained_ensemble_model.pkl'
        else:
            yaml_info['prediction_model'] = 'pretrained_'+models[0]['name']+'_model.pkl'

        yaml_info['features'] = features
        yaml_info['training'] = 'running'
        yaml_path = os.path.join("models",model_version, 'model.yaml')
        with open(yaml_path, 'w') as output:
            yaml.dump(yaml_info, output)

        estimators=[]
        weights = []
        activation_functions = []
        ensemble = False
        if yaml_info['prediction_model'] == 'pretrained_ensemble_model.pkl':
            ensemble = True

        for model in models:
            print(model)
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

            estimators.append((model['name'],model_classifier))
            weights.append(int(model['weight']))
            label_data= get_info(conf_rep)

            yaml_info[model['name']] = dict()
            yaml_info[model['name']]['eval_accuracy'] = float(eval_accuracy)
            yaml_info[model['name']]['test_score'] = float(test_score)
            yaml_info[model['name']]['conf_rep'] = label_data
            yaml_info[model['name']]['weight'] = model['weight']

        if yaml_info['prediction_model'] == 'pretrained_ensemble_model.pkl':
            print(estimators,weights,features,model_version)
            train_ensemble(estimators,weights,features,model_version)

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
        letters = pre_process_image(image)
        # for letter in letters:
        #     cv.imshow('image', letter)
        #     cv.waitKey(0)
        print('Number Of Letters: '+ str(len(letters)))
        print(model_version)
        yaml_path = os.path.join(f"models/{model_version}", "model.yaml")
        with open(yaml_path, 'r') as f:
            yaml_info = yaml.safe_load(f)
            features = yaml_info['features']
            character_features = get_character_features(features, letters)
            case_features = get_case_features(letters)
            if model_version != 'topPerformer':
                model_name = yaml_info['prediction_model']
                model = pickle.load(open(os.path.join(f"models/{model_version}", model_name), 'rb' ))
            word = ''
            for i in range(len(character_features)):
                if model_version == 'topPerformer':
                    prediction = predict_model(character_features[i], case_features[i])
                else:
                    prediction = model.predict(character_features[i])
                print(prediction)
                word = word + prediction[0]
            # print(prediction)
        #     return render_template("predict.html")
    response = jsonify(word = word)
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
            continue
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