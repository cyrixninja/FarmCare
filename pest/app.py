import os
from unittest import result
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from PIL import Image, ImageOps

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()

# You can also use pretrained model from Keras
# Check https://keras.io/applications/

model = 'model.h5'
print('Model loaded. Check http://127.0.0.1:5000/')


def classifier(img, file):
    np.set_printoptions(suppress=True)
    model = keras.models.load_model(file)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = img
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    prediction = model.predict(data)
    return prediction


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        image = Image.open(file_path).convert('RGB')
        # Make prediction
        prediction = classifier(image, model)

        # x = x.reshape([64, 64]);
        pest_class = ['Aphids',
                      'Armyworm',
                      'Beetle',
                      'Bollyworm',
                      'Grasshopper',
                      'Mites',
                      'Mosquito',
                      'Sawfly',
                      'Stem_Borer'
                      ]
        a = prediction[0]
        ind = np.argmax(a)
        print('Prediction:', pest_class[ind])
        result1 = pest_class[ind]
        if result1 == "Aphids":
            result = """
            They are Aphids.
            Learn about them here - https://www.almanac.com/pest/aphids
            """
        elif result1 == "Armyworm":
            result = """
            They are Armyworms.
            Learn about them here - https://www.planetnatural.com/pest-problem-solver/garden-pests/armyworm-control/
            """
        elif result1 == "Beetle":
            result = """
            They are Beetles.
            Learn about them here - https://smithspestmanagement.com/blog/post/how-to-get-rid-of-beetles-in-your-home-yard/
            """
        elif result1 == "Bollyworm":
            result = """
            They are Bollyworm.
            Learn about them here - https://plantix.net/en/library/plant-diseases/600161/pink-bollworm
            
            """
        elif result1 == "Grasshopper":
            result = """
            They are Grasshopper.
            Learn about them here - https://www.masterclass.com/articles/how-to-get-rid-of-grasshoppers-explained#what-attracts-grasshoppers-to-gardens
            
            """
        elif result1 == "Mites":
            result = """
            They are Mites.
            Learn about them here - https://www.gardeningknowhow.com/plant-problems/pests/insects/spider-mite-control.htm
            
            """
        elif result1 == "Mosquito":
            result = """
            They are Mosquito.
            
            """
        elif result1 == "Sawfly":
            result = """
            They are Sawfly.
            Learn about them here - https://www.planetnatural.com/pest-problem-solver/tree-pests/pear-sawfly-control/
            
            """
        elif result1 == "Stem_Borer":
            result = """
            They are Stem_Borer.
            Learn about them here - http://www.knowledgebank.irri.org/training/fact-sheets/pest-management/insects/item/stem-borer
            
            """
        return result
    return None


if __name__ == '__main__':
    # app.run(port=5002, debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
    app.run()
