# Dependencies
from flask import Flask, request, jsonify
import traceback
import flask
import numpy as np
from tensorflow import keras
import tensorflow
from PIL import Image
import shutil
import requests
from io import BytesIO
import urllib.parse
import io

print(tensorflow.__version__)
# print(flask.__version__)
# print(np.__version__)
# print(tensorflow.__version__)pip install --upgrade tensorflow
# print(keras.__version__)

# Your API definition
app = Flask(__name__)
model = keras.models.load_model(r"updated_model.hdf5")
print('Model loaded')


@app.route('/')
def predict():
    if model:
        try:
            url1 = request.args['url']
            url1 = url1[84:]
            url1 = url1[:33]
            url1 = url1[:-10]
            # print(url1)
            #print("hi there bro whatsup")
            encoded_url1 = urllib.parse.quote(url1, safe="")
            #print("URL part 1 is : " + encoded_url1)
            url2 = request.args['token']
            encoded_url2 = urllib.parse.quote(url2, safe="")
            #print("URL part 2 is : " + encoded_url2)
            url = encoded_url1 + "?alt=media&token=" + encoded_url2
            url = "https://firebasestorage.googleapis.com/v0/b/smartinfantcradle-api.appspot.com/o/rpi%2F" + url
            #print("Final URL is: " + url)
            #url = "https://i.pinimg.com/550x/4b/e1/ba/4be1baf30d5f4c9132025ff7697dbc5b.jpg"
            #response = requests.get(url)
            res = requests.get(url, stream=True)

            if res.status_code == 200:
                with open('temp.jpg', 'wb') as f:
                    shutil.copyfileobj(res.raw, f)
                print('Image sucessfully Downloaded: ', 'temp.jpg')
            else:
                print('Image Couldn\'t be retrieved')

            img = Image.open("temp.jpg")
            print("image printed")
            img = img.resize((200, 200))
            img = np.expand_dims(img, axis=0)
            prediction = model.predict(img)
            maxP = max(prediction[0])
            print(maxP)
            result = "crying"
            if prediction[0][1] == maxP:
                result = "happy"
            elif prediction[0][2] == maxP:
                result = "sleeping"
            print(result)
            return jsonify(prediction=result)

        except:
            return jsonify({'trace': traceback.format_exc()})
    else:
        print('Train the model first')
        return ('No model here to use')


if __name__ == '__main__':
    app.run(debug=True)
