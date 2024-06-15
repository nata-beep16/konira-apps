import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import gdown
import joblib
import jwt
import numpy as np
import pandas as pd
from functools import wraps
from http import HTTPStatus
from PIL import Image
from flask import Flask, jsonify, request
from google.cloud import storage
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as tf_image
from zipfile import ZipFile 

load_dotenv()

app = Flask(__name__)

app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg'])
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['MODEL_KONIRA_CLASSIFICATION'] = './models/modelLovenBaru.h5'
app.config['GOOGLE_APPLICATION_CREDENTIALS'] = './credentials/gcs.json'

model_classification = load_model(app.config['MODEL_KONIRA_CLASSIFICATION'], compile=False)

bucket_name = os.environ.get('BUCKET_NAME', 'konira-bucket')
client = storage.Client.from_service_account_json(json_credentials_path=app.config['GOOGLE_APPLICATION_CREDENTIALS'])
bucket = storage.Bucket(client, bucket_name)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']
           

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'Message': 'KONIRA Apps',
    }), HTTPStatus.OK

@app.route('/predict', methods=['POST'])
def predict_konira_classification():
    if request.method == 'POST':
        reqImage = request.files['image']
        if reqImage and allowed_file(reqImage.filename):
            filename = secure_filename(reqImage.filename)
            reqImage.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            img = Image.open(image_path).convert('RGB')
            img = img.resize((224, 224)) # image size
            x = tf_image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = x / 255

            #predict model
            classificationResult = model_classification.predict(x, batch_size=1) 
            class_list = ['miner', 'modisease', 'phoma', 'rust']
            classification_class = class_list[np.argmax(classificationResult[0])]

            image_name = image_path.split('/')[-1]
            blob = bucket.blob('images/' + image_name)
            blob.upload_from_filename(image_path) 
            os.remove(image_path)
            return jsonify({
                'status': {
                    'code': HTTPStatus.OK,
                    'message': 'Success predicting',
                    'data': { 'class': classification_class  }
                }
            }), HTTPStatus.OK 
        else:
            return jsonify({
                'status': {
                    'code': HTTPStatus.BAD_REQUEST,
                    'message': 'Invalid file format. Please upload a JPG, JPEG, or PNG image.'
                }
            }), HTTPStatus.BAD_REQUEST
    else:
        return jsonify({
            'status': {
                'code': HTTPStatus.METHOD_NOT_ALLOWED,
                'message': 'Method not allowed'
            }
        }), HTTPStatus.METHOD_NOT_ALLOWED

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5001)))