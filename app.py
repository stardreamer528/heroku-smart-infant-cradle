# Dependencies
from flask import Flask, request, jsonify
import traceback
import flask
import numpy as np
from tensorflow import keras
import tensorflow
from PIL import Image
import requests
from io import BytesIO
import urllib.parse


print(flask.__version__)
print(np.__version__)
print(tensorflow.__version__)
print(keras.__version__)

# Your API definition
app = Flask(__name__)
model = keras.models.load_model('best_model.hdf5')
print('Model loaded')


@app.route('/')
def predict():
    if model:
        try:
            url1 = request.args['url']
            url1 = url1[84:]
            url1 = url1[:22]
            print(url1)
            encoded_url1 = urllib.parse.quote(url1, safe="")
            print("URL part 1 is : " + encoded_url1)
            url2 = request.args['token']
            encoded_url2 = urllib.parse.quote(url2, safe="")
            print("URL part 2 is : " + encoded_url2)
            url = encoded_url1 + "?alt=media&token=" + encoded_url2
            url = "https://firebasestorage.googleapis.com/v0/b/smartinfantcradle-api.appspot.com/o/home" + url
            print("Final URL is: " + url)
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))

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

            return jsonify({'prediction': result})

        except:
            return jsonify({'trace': traceback.format_exc()})
    else:
        print('Train the model first')
        return ('No model here to use')


if __name__ == '__main__':
    app.run(debug=True)
