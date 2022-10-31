import pickle
from datetime import datetime
from genericpath import exists
from os import makedirs
import os
from flask import Flask, jsonify, render_template, request, redirect, url_for
import yaml
import cv2 as cv
from features import get_character_features
from knn_classifier import train_knn
from svm_classifier import train_svm
from decision_tree_classifier import train_dt

app = Flask(__name__)

@app.route('/info', methods = ['GET'])
def get_available_models():
    """
    This function searches for all models 
    and returns its characterstics
    """
    models = os.listdir('models')
    response = jsonify(models)
    response.headers.add("Access-Control-Allow-Origin", "*")

    return response
@app.route('/train_new_model', methods= ['GET','POST'])
def train_new_model():
    if request.method == 'POST':
        features = request.form['features']
        models = request.form['models']
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
    if request.method == 'POST':
        model_version = request.form.get('model_version')
        yaml_path = os.path.join(f"models/{model_version}", "model.yaml")
        # character = request.form.get('character') # this should be converted to numpy array if it isn't
        character = cv.imread(os.path.join("processed_images","img001-001.png"))
        with open(yaml_path, 'r') as f:
            yaml_info = yaml.safe_load(f)
            features = yaml_info['features']
            character_features = get_character_features(features, character)
            model_name = yaml_info['prediction_model']
            model = pickle.load(open(os.path.join(f"models/{model_version}", model_name), 'rb' ))
            prediction = model.predict(character_features)
            return render_template("predict.html")
    return render_template("predict.html")


if __name__ == "__main__":
    app.run(debug = True)