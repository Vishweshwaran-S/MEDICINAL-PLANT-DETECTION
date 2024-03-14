from flask import Flask, request, render_template, redirect, url_for
import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('C:/data/gen ai projects/leaf_detection.h5')  # Adjust path if necessary


class_labels = {
    0: 'Eucalyptus',
    1: 'Neem',
    2: 'Tulsi',
    3: 'Mint',
    4: 'Aloe Vera',
    5: 'Turmeric'
}

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_and_predict():
    if 'image' not in request.files:
        return redirect(request.url)  

    image_file = request.files['image']
    if image_file.filename == '':
        return redirect(request.url)  

    
    img = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (220, 220))  
    img_normalized = img.astype('float32') / 255.0 

    
    prediction = model.predict(np.expand_dims(img_normalized, axis=0))[0]
    predicted_class = class_labels[np.argmax(prediction)]

    return render_template('result.html', prediction=predicted_class, image_file=image_file.filename)
if __name__ == '__main__':
    app.run(debug=True)  

