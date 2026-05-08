import os
import cv2
import numpy as np
import tensorflow as tf

# =========================
# LOAD MODEL ONCE
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "models", "anti_spoof_model.h5")
model = tf.keras.models.load_model(MODEL_PATH)

# =========================
# CONFIG
# =========================
IMG_SIZE = (224, 224)
THRESHOLD = 0.75


# =========================
# PREPROCESS FUNCTION
# =========================
def preprocess(frame):
    img = cv2.resize(frame, IMG_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img


# =========================
# MAIN FUNCTION
# =========================
def is_real_face(frame):
    """
    Input: frame (BGR image from OpenCV)
    Output: (bool, confidence)
    """

    img = preprocess(frame)

    prediction = model.predict(img, verbose=0)[0][0]

    if prediction >= THRESHOLD:
        return True, float(prediction)
    else:
        return False, float(prediction)
    
print("Model path:", MODEL_PATH)

if __name__ == "__main__":
    import cv2

    cap = cv2.VideoCapture(1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        real, conf = is_real_face(frame)

        print("REAL" if real else "SPOOF", conf)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()