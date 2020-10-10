import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import cv2
import os
from keras.preprocessing import image
from keras.models import model_from_json


app = Flask(__name__)

UPLOAD_FOLDER = './static/images'


app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('index.html',user_image=os.path.join(app.config['UPLOAD_FOLDER'], 'logo.png'))

@app.route('/upload',methods=['POST'])
def upload():
    try:
        
        json_file = open('modelg.json','r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights("modelg.h5")
        loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        label2gar={0:"Cardboard",
               1:"Glass",
               2:"Metal",
               3:"Paper",
               4:"Plastic",
               5:"Trash"}



        # check if the post request has the file part
        file = request.files['myImage']
        nm=file.filename

        filename =file.filename
        path=os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)


        
        image_path=path
        img=image.load_img(image_path,target_size=(40,40))
        img_array=image.img_to_array(img)
        img_array=np.array(img_array)
        img_array=img_array.reshape(40,40,3)
        prob=loaded_model.predict(img_array.reshape(-1,40,40,3))
        prob=np.array(prob)
        

        return render_template('index.html', status=label2gar[np.argmax(prob)],user_image=path)

    except Exception as err:
        print("Error occurred")
        return render_template('index.html', status=err)

    
if __name__ == "__main__":
    app.run(debug=True)