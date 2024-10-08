import cv2
import numpy as np
import os

def load_images(image_dir, labels, size=(128, 128)):
    images = []
    for img_file in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_file)
        img = cv2.imread(img_path)
        img = cv2.resize(img, size)
        images.append(img)
    return np.array(images), np.array(labels)

def normalize_images(images):
    return images / 255.0

