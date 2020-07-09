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


UPLOAD_FOLDER = './image'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def welcome():
    return "Hello World!"

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
            glcm_one_image(file)
            return json.dumps({ 'message' : 'Success' })

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
    df['filename'] = filename
    df['contrast'] = contrast
    df['energy'] = energy
    df['homo'] = homo
    df['corr'] = corr
    df['label'] = label

    df.to_csv('result/csv_test/test_csv.csv')


def glcm(filename):
    contrast = []
    energy = []
    homo = []
    corr = []
    filename = []
    label = []

    mypath = './image/'
    kelas = [f for f in os.listdir(mypath)]
    kelas = kelas[1:]
    for klas in kelas:
        files = [f for f in os.listdir(mypath + klas + '/')]
        print(files)
        for file in files:
            img = cv.imread(mypath + klas + '/' + file)
            filename.append(file)
            label.append(klas)

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
            cv.imwrite('result/' + klas +'/' + file, gray)


            
            # ===== GLCM ====
            glcm = greycomatrix(gray, [5], [0], 256, symmetric=True, normed=True)
            contrast.append(greycoprops(glcm, 'contrast')[0, 0])
            energy.append(greycoprops(glcm, 'energy')[0, 0])
            homo.append(greycoprops(glcm, 'homogeneity')[0, 0])
            corr.append(greycoprops(glcm, 'correlation')[0, 0])


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=105)