from flask import Flask, render_template,request
import keras
from PIL import Image
import os


import pandas as pd


import numpy as np
import tensorflow as tf


from keras.models import load_model



app = Flask(__name__)
IMG_WIDTH = 100
IMG_HEIGHT = 75
model = load_model('./test.h5')

@app.route("/",methods=['GET'])
def hello():
    return render_template('template.html')

@app.route("/",methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    imagepath = "./images/" + imagefile.filename
    imagefile.save(imagepath)

    image = tf.keras.utils.load_img(imagepath, target_size=(75,100))
    image = tf.keras.utils.img_to_array(image)
    image=  image.reshape((1,image.shape[0],image.shape[1],image.shape[2]))
    t = model.predict(image)
    t = t.flatten()

    t = pd.Series(t).to_json(orient='values',double_precision=4)




    return render_template('template.html',prediction =t)




if (__name__ == "__main__"):
    app.run()
