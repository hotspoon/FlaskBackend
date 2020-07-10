import os
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
import json
import numpy as np
import cv2 as cv
from os import listdir
import pandas as pd
import re
from skimage.feature import greycomatrix
from skimage.feature import greycoprops
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from flask_swagger import swagger
from flask import Flask, jsonify
from flask_swagger_ui import get_swaggerui_blueprint


UPLOAD_FOLDER = './image'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


SWAGGER_URL = '/api/docs'  # URL for exposing Swagger UI (without trailing '/')
API_URL = '/static/swaggers.json'  # Our API url (can of course be a local resource)

# Call factory function to create our blueprint
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,  # Swagger UI static files will be mapped to '{SWAGGER_URL}/dist/'
    API_URL,
    config={  # Swagger UI config overrides
        'app_name': "Classification Tobacou"
    },
    # oauth_config={  # OAuth config. See https://github.com/swagger-api/swagger-ui#oauth2-configuration .
    #    'clientId': "your-client-id",
    #    'clientSecret': "your-client-secret-if-required",
    #    'realm': "your-realms",
    #    'appName': "your-app-name",
    #    'scopeSeparator': " ",
    #    'additionalQueryStringParams': {'test': "hello"}
    # }
)
# Register blueprint at URL
# (URL must match the one given to factory function above)
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def welcome():
    return "jsonify(swagger(app))"

@app.route('/classification', methods=['POST'])
def classification():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return json.dumps({ 'message' : 'No selected file' })
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            return json.dumps({ 'message' : 'No selected file' })
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            path_result_test = glcm_one_image(file)
            result = runSvm(path_result_test)
            return json.dumps({ 'message' : 'Success', 'label' : result[0] })

def glcm_one_image(file):
    contrast = []
    energy = []
    homo = []
    corr = []
    filename = []
    label = []

    img = cv.imread(UPLOAD_FOLDER + '/' + file.filename)
    print(img)
    filename.append(file.filename)
    label.append(None)

    # ==== resize image =======
    print(file)
    dim = (96, 96)
    img = cv.resize(img, dim, interpolation = cv.INTER_AREA)

    # ==== remove background grabcut=====
    mask = np.zeros(img.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    rect = (1,1,290,290)
    cv.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img = img*mask2[:,:,np.newaxis]

    # ==== grayscale ====
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # ==== save new image =====
    cv.imwrite('result/image_gray/' + file.filename, gray)

    # ===== GLCM ====
    glcm = greycomatrix(gray, [5], [0], 256, symmetric=True, normed=True)
    contrast.append(greycoprops(glcm, 'contrast')[0, 0])
    energy.append(greycoprops(glcm, 'energy')[0, 0])
    homo.append(greycoprops(glcm, 'homogeneity')[0, 0])
    corr.append(greycoprops(glcm, 'correlation')[0, 0])

    df = pd.DataFrame()
    df['contrast'] = contrast
    df['energy'] = energy
    df['homo'] = homo
    df['corr'] = corr

    df.to_csv('result/csv_test/'+ file.filename +'.csv')

    return 'result/csv_test/'+ file.filename +'.csv'

def runSvm(test_csv):
    test_result = pd.read_csv(test_csv)
    X_test_result = test_result[['contrast', 'energy', 'homo', 'corr']]
    print(X_test_result)

    df = pd.read_csv('result/csv_train/result_features.csv')

    X = df[['contrast', 'energy', 'homo', 'corr']]
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05)
    print(X_train)

    svclassifier = SVC(kernel='linear')
    svclassifier.fit(X_train, y_train)

    y_pred = svclassifier.predict(X_test_result)

    print(y_pred)
    # print(confusion_matrix(y_test,y_pred))
    # print(classification_report(y_test,y_pred))

    return y_pred

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=105)