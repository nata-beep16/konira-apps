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

gdown.download('https://drive.google.com/uc?id=1LSjqv4GteWiC7dSTY_2e1TggvglgeXR4')
with ZipFile('./models.zip', 'r') as modelFolder: 
    modelFolder.extractall()

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

SECRET_KEY = os.environ.get('SECRET_KEY')
if SECRET_KEY is None:
    print("SECRET_KEY not found in environment variables.")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']
           
def token_required(f):
    @wraps(f)
    def decorator(*args, **kwargs):
        token = request.headers.get('Authorization', None)
        if not token:
            return jsonify({'message': 'Invalid token'}), 401
        try:
            token_prefix, token_value = token.split()
            if token_prefix.lower() != 'bearer':
                raise ValueError('Invalid token prefix')
            data = jwt.decode(token_value, SECRET_KEY, algorithms=['HS256'])
        except jwt.ExpiredSignatureError:
            return jsonify({'message': 'Token has expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'message': 'Invalid token'}), 401
        except ValueError:
            return jsonify({'message': 'Invalid token format'}), 401
        return f(data, *args, **kwargs)
    return decorator

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'Message': 'KONIRA Apps',
    }), HTTPStatus.OK

@app.route('/predict', methods=['POST'])
@token_required
def predict_konira_classification(data):
    if data is None:
        return jsonify({
            'status': {
                'code': HTTPStatus.FORBIDDEN,
                'message': 'Access denied',
            }
        }), HTTPStatus.FORBIDDEN
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

        #     predicted_class = None
        #     if(classification_class == 'Ikan'):
        #         classes = model_marine_grading_fish.predict(x, batch_size=1) 
        #         class_list = ['A', 'B', 'C']
        #         predicted_class = class_list[np.argmax(classes[0])]
        #     elif(classification_class == 'Udang'):
        #         classes = model_marine_grading_shrimp.predict(x, batch_size=1) 
        #         class_list = ['A', 'B', 'C']
        #         predicted_class = class_list[np.argmax(classes[0])]
        #    else:
        #         predicted_class = 'Grade tidak tersedia' #====ubah====
            image_name = image_path.split('/')[-1]
            blob = bucket.blob('images/' + image_name)
            blob.upload_from_filename(image_path) 
            os.remove(image_path)
            return jsonify({
                'status': {
                    'code': HTTPStatus.OK,
                    'message': 'Success predicting',
                    'data': { 'class': classification_class, 'grade': predicted_class }
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