from copyreg import pickle
from datetime import datetime
from genericpath import exists
from http.client import OK
from os import makedirs
import os
from flask import Flask
from requests import request
import yaml
from Character_Detection.features import get_character_features

from Character_Detection.knn_classifier import train_knn

app = Flask(__name__)

@app.route('/', methods = ['GET'])
def get_available_models():
    """
    This function searches for all models 
    and returns its characterstics
    """
    return ''

@app.route('/train_new_model', methods= ['POST'])
def train_new_model():
    features = request.form['features']
    models = request.form['models']
    model_version = datetime.now().replace('-', '_').replace(' ','_').replace(':','_')
    makedirs(model_version)
    yaml_info = dict()
    if len(models)> 1:
        yaml_info['prediction_model'] = 'ensemble.pkl'
    else:
        yaml_info['prediction_model'] = models[0]['name']
    yaml_info['training'] = 'running'
    yaml_path = os.path.join(model_version, 'model.yaml')
    with open(yaml_path, 'w') as output:
        yaml.dump(yaml_info, output)


    for model in models:
        if model['name'] == 'knn':
            train_knn(features, model_version)
        if model['name'] == 'svm':
            train_knn(features, model_version)
        if model['name'] == 'dt':
            train_knn(features, model_version)
    
    with open(yaml_path, 'r') as f:
        yaml_info = yaml.load(f)
        yaml_info['training'] = 'completed'
    with open(yaml_path, 'w') as output:
        yaml.dump(yaml_info, output)

    
@app.route('/predict', methods=['GET'])
def predict():
    model_version = request.form.get('model_version')
    yaml_path = os.path.join(model_version, 'model.yaml')
    character = request.form.get('character') # this should be converted to numpy array if it isn't
    with open(yaml_path, 'r') as f:
        yaml_info = yaml.load(f)
        features = yaml_info['features']
        character_features = get_character_features(features, character)
        model_name = yaml_info['prediction_model']
        model_path = os.path.join(model_version, model_name)
        model = pickle.load(open(model_name, 'rb' ))
        prediction = model.predict(character_features)
        return prediction


if __name__ == "__main__":
    app.run(debug = True)