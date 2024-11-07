import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, request, redirect, url_for, flash
from src.predict import predict_image

app = Flask(__name__)
app.secret_key = 'b3b8e8e5f90746f9b98cf8c4ebd1c4d8'

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../models/cnn_model.h5")
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/test")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    result = request.args.get('result') 
    return render_template('index.html', result=result)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        flash('No file part') 
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        flash('No selected file')  
        return redirect(request.url)

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    
    file.save(file_path)
    
    prediction = predict_image(file_path, MODEL_PATH)
    result = f"Predicted Nitrogen Content: {prediction:.2f} mg/g"

    return redirect(url_for('index', result=result))  

if __name__ == '__main__':
    app.run(debug=True)