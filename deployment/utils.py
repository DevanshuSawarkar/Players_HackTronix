from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'

def gen_labels():
    train = './dataset/TRAIN'
    train_generator = ImageDataGenerator(rescale=1/255)

    train_generator = train_generator.flow_from_directory(
        train,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary'
    )
    labels = {0: 'Organic', 1: 'Recyclable'}  # Adjusted for binary classification
    return labels

def preprocess(image):
    image = image.resize((224, 224))  # Remove Image.ANTIALIAS
    image = np.array(image, dtype='float32') / 255.0
    return image

def model_arc():
    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = False
    
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
