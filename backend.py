import pickle
from datetime import datetime
from genericpath import exists
from os import makedirs
import os
from flask import Flask, jsonify, render_template, request, redirect, url_for
from matplotlib.style import available
import yaml
import cv2 as cv
from features import get_character_features
from features import pre_process_image
from knn_classifier import train_knn
from svm_classifier import train_svm
from decision_tree_classifier import train_dt
from flask_cors import CORS
import base64
from PIL import Image
import io
import numpy as np

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
        yaml_model_path = os.path.join('models', model, 'model.yaml')
        yaml_info = read_yaml(yaml_model_path)
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
        model_version = str(datetime.now()).replace('-', '_').replace(' ','_').replace(':','_')
        makedirs("models/"+model_version)
        yaml_info = dict()
        if len(models)> 1:
            yaml_info['prediction_model'] = 'ensemble.pkl'
        else:
            yaml_info['prediction_model'] = models[0]['name']
        yaml_info['training'] = 'running'
        yaml_path = os.path.join("models",model_version, 'model.yaml')
        with open(yaml_path, 'w') as output:
            yaml.dump(yaml_info, output)
   
        for model in models:
            if model['name'] == 'knn':
                eval_accuracy, test_score, conf_rep = train_knn(model['features'], model_version)
            if model['name'] == 'svm':
                eval_accuracy, test_score, conf_rep = train_svm(model['features'], model_version)
            if model['name'] == 'dt':
                eval_accuracy, test_score, conf_rep = train_dt(model['features'], model_version)

        label_data= get_info(conf_rep)

        with open(yaml_path, 'r') as f:
            yaml_info = yaml.safe_load(f)
            yaml_info['training'] = 'completed'
            yaml_info['eval_accuracy'] = float(eval_accuracy)
            yaml_info['test_score'] = float(test_score)
            yaml_info['conf_rep'] = label_data
        with open(yaml_path, 'w') as output:
            yaml.dump(yaml_info, output)

    response = jsonify(training='completed', eval_accuracy=eval_accuracy, test_score=test_score)
    print(response.json)
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
        image = pre_process_image(image)
        # cv.imshow('image', image)
        # cv.waitKey(0)
        yaml_path = os.path.join(f"models/{model_version}", "model.yaml")
        with open(yaml_path, 'r') as f:
            yaml_info = yaml.safe_load(f)
            features = yaml_info['features']
            character_features = get_character_features(features, image)
            model_name = yaml_info['prediction_model']
            model = pickle.load(open(os.path.join(f"models/{model_version}", model_name), 'rb' ))
            prediction = model.predict(character_features)
            print(prediction)
        #     return render_template("predict.html")
    response = jsonify(request.json)
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