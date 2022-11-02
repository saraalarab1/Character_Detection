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


@app.route('/train_new_model', methods= ['GET','POST'])
def train_new_model():
    if request.method == 'POST':
        print(request.json)
        features = []
        models = []
        model_version = str(datetime.now()).replace('-', '_').replace(' ','_').replace(':','_')
        makedirs("models/"+model_version)
        yaml_info = dict()
        if len(models)> 1:
            yaml_info['prediction_model'] = 'ensemble.pkl'
        else:
            yaml_info['prediction_model'] = models[0]
        yaml_info['training'] = 'running'
        yaml_path = os.path.join("models",model_version, 'model.yaml')
        with open(yaml_path, 'w') as output:
            yaml.dump(yaml_info, output)
        print(models)
        models = [models]
        features = [features]
        for model in models:
            if model == 'knn':
                train_knn(features, model_version)
            if model == 'svm':
                train_svm(features, model_version)
            if model == 'dt':
                train_dt(features, model_version)
        
        with open(yaml_path, 'r') as f:
            yaml_info = yaml.load(f)
            yaml_info['training'] = 'completed'
        with open(yaml_path, 'w') as output:
            yaml.dump(yaml_info, output)

    return render_template("train_new_model.html")

    
@app.route('/predict', methods=['GET','POST'])
def predict():
    print('predicting')
    if request.method == 'POST':
        base64_image = request.json['image']
        base64_image = base64_image.split('base64,')[1]
        im_bytes = base64.b64decode(base64_image)
        im_arr = np.frombuffer(im_bytes, dtype=np.uint8)  # im_arr is one-dim Numpy array
        img = cv.imdecode(im_arr, flags=cv.IMREAD_COLOR)
        model_version = request.json['model_version']
        image = pre_process_
        # yaml_path = os.path.join(f"models/{model_version}", "model.yaml")
        # # character = request.form.get('character') # this should be converted to numpy array if it isn't
        # character = cv.imread(os.path.join("processed_images","img001-001.png"))
        # with open(yaml_path, 'r') as f:
        #     yaml_info = yaml.safe_load(f)
        #     features = yaml_info['features']
        #     character_features = get_character_features(features, character)
        #     model_name = yaml_info['prediction_model']
        #     model = pickle.load(open(os.path.join(f"models/{model_version}", model_name), 'rb' ))
        #     prediction = model.predict(character_features)
        #     return render_template("predict.html")
    response = jsonify(request.json)
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response





# Take in base64 string and return PIL image
def stringToImage(base64_string):
    imgdata = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(imgdata))

# convert PIL Image to an RGB image( technically a numpy array ) that's compatible with opencv
def toRGB(image):
    return cv.cvtColor(np.array(image), cv.COLOR_BGR2RGB)

if __name__ == "__main__":
    app.run(debug = True)