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


print(flask.__version__)
print(np.__version__)
print(tensorflow.__version__)
print(keras.__version__)

# Your API definition
app = Flask(__name__)
model = keras.models.load_model("best_model.hdf5")
print('Model loaded')


@app.route('/')
def predict():
    if model:
        try:
            url = request.args['url']  # user provides url in query string
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
