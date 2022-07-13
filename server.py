import os
import keras.utils as ku
import numpy as np
from flask import Flask, render_template, request
from keras.models import load_model

# df=pd.read_csv("prac")
from werkzeug.utils import secure_filename

app = Flask(__name__)
# deserializing to read the file

# Model saved with Keras model.save()
MODEL_PATH = 'model_vgg19.h5'

# Load your trained model
model = load_model(MODEL_PATH)


def model_predict(img_path, model, preds=None):
    img = ku.load_img(img_path, target_size=(300, 300))

    # Preprocessing the image
    x = ku.img_to_array(img)
    img_test = np.expand_dims(x, axis=0)

    classes = model.predict(img_test)

    print(classes)

    values = classes[0]
    index1 = np.argmax(values)
    print(index1)
    if index1 == 0:
        preds='paper'
    elif index1 == 1:
        preds='rock'
    else:
        preds='scissors'
    return preds

#model_predict('Dataset/validation/paper8.png', model)

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
            # Make prediction
            preds = model_predict(file_path, model)
            result = preds
            return render_template('index.html',pred='{}'.format(result))
        return None

if __name__ == '__main__':
    app.run(debug=True)