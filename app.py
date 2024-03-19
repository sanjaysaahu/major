from flask import Flask, render_template, request, send_from_directory
from flask_uploads import UploadSet, configure_uploads, IMAGES
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
from keras.preprocessing.image import load_img, img_to_array
import cv2
import imgaug.augmenters as iaa
import os
import matplotlib.pyplot as plt


import numpy as np

def dice_coef(y_true, y_pred):
    y_truef=K.flatten(y_true)
    y_predf=K.flatten(y_pred)
    And=K.sum(y_truef* y_predf)
    return((2* And + 100) / (K.sum(y_truef) + K.sum(y_predf) + 100))

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def iou(y_true, y_pred):
    intersection = K.sum(y_true * y_pred)
    sum_ = K.sum(y_true + y_pred)
    jac = (intersection + 100) / (sum_ - intersection + 100)
    return jac

def jac_distance(y_true, y_pred):
    y_truef=K.flatten(y_true)
    y_predf=K.flatten(y_pred)
    return 1-iou(y_truef, y_predf)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'

# Configure the image uploads
photos = UploadSet("photos", IMAGES)
app.config["UPLOADED_PHOTOS_DEST"] = "uploads"
app.config["STATIC_FOLDER"] = "static"
configure_uploads(app, photos)

model = load_model("unet_inception.h5", custom_objects={'dice_coef_loss': dice_coef_loss, 'iou': iou, 'dice_coef': dice_coef})
aug = iaa.Sharpen(alpha=(1.0), lightness=(1.5))

def predict_segmentation(image_path,result_path):
    image1 = cv2.imread(image_path)
    image1 = cv2.resize(image1, (256, 256))
    image = aug.augment_image(image1)
    image = image[:, :, 1]
    image[image < 0.2] = 0.5
    image = image / 255
    predicted = model.predict(image[np.newaxis, :, :])
    predicted[predicted < 0.25] = 0
    img = predicted[0,:,:,0]
    img1 = (predicted[0, :, :, 0] * 255).astype(np.uint8)
    cv2.imwrite(result_path, img1)
    mean,std=cv2.meanStdDev(img)
    pixels = cv2.countNonZero(img)
    image_area = img.shape[0] * img.shape[1]
    area_ratio = (pixels / image_area) * 100
    img = img*255
    img[img<1]=1
    img[img>100]=255
    M= cv2.moments(img)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return {
        'mean': mean[0][0],
        'std': std[0][0],
        'area_ratio': area_ratio,
        'cX': cX,
        'cY': cY
    }


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('index.html', message='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', message='No selected file')

    if file:
        # Save the uploaded file
        file_path = app.config['UPLOAD_FOLDER'] + '/' + file.filename
        file.save(file_path)

        # Perform segmentation using the pre-trained model
        result_path = app.config['RESULT_FOLDER'] + '/result_' + file.filename
        info = predict_segmentation(file_path, result_path)

        return render_template('index.html', message='File uploaded successfully', filename=file.filename,info=info)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/<filename>')
def result_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

if __name__ == '__main__':
    app.run()