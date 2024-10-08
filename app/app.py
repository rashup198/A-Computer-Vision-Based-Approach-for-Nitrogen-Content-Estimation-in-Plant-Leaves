import sys
import os
from flask import Flask, render_template, request, redirect, url_for, flash
from src.predict import predict_image

app = Flask(__name__)
app.secret_key = 'b3b8e8e5f90746f9b98cf8c4ebd1c4d8'  # Set the secret key here
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../models/cnn_model.h5")
UPLOAD_FOLDER = 'path/to/upload/folder'  # Make sure to define UPLOAD_FOLDER

@app.route('/')
def index():
    result = request.args.get('result')  # Get the result from the query string
    return render_template('index.html', result=result)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        flash('No file part')  # Flash message for no file part
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        flash('No selected file')  # Flash message for no selected file
        return redirect(request.url)

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    
    # Save the uploaded file
    file.save(file_path)
    
    prediction = predict_image(file_path, MODEL_PATH)
    result = f"Predicted Nitrogen Content: {prediction:.2f}"

    return redirect(url_for('index', result=result))  # Redirect with result


if __name__ == '__main__':
    app.run(debug=True)
