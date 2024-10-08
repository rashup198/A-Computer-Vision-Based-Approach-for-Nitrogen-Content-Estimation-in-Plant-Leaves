import cv2
import numpy as np
import tensorflow as tf

def predict_image(image_path, model_path):
    model = tf.keras.models.load_model(model_path)
    
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    
    prediction = model.predict(img)
    return prediction[0][0]

if __name__ == "__main__":
    image_path = "../data/test/sample_leaf.jpg"
    model_path = "../models/cnn_model.h5"
    predicted_nitrogen = predict_image(image_path, model_path)
    print(f"Predicted Nitrogen Content: {predicted_nitrogen:.2f}")
