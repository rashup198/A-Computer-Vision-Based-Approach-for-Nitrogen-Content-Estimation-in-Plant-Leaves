import preprocess
import model
from sklearn.model_selection import train_test_split
import tensorflow as tf
import os

image_dir = "../data/trains"

image_files = os.listdir(image_dir)
print("this is",image_files)

labels = [float(img_file.split('.')[0]) for img_file in image_files]  

images, labels = preprocess.load_images(image_dir, labels) 
images = preprocess.normalize_images(images)

X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

input_shape = (128, 128, 3)
cnn_model = model.create_cnn_model(input_shape)

cnn_model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val))

model_save_path = "../models/cnn_model.h5"
cnn_model.save(model_save_path)
print(f"Model saved at {model_save_path}")
