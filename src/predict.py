import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "anti_spoof_model.h5")

model = tf.keras.models.load_model(MODEL_PATH)

img_path = input("Enter image path: ")

img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

prediction = model.predict(img_array)[0][0]

if prediction >= 0.5:
    print(f"REAL FACE ✅  Confidence: {prediction:.2f}")
else:
    print(f"SPOOF FACE ❌ Confidence: {1-prediction:.2f}")