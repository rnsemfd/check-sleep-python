import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

model = load_model('models/test_model.h5')

export_path =r'C:\Users\rnsem\Desktop\eye_blink_detector-master\eye_blink_detector-master\test'
model.save(export_path)
