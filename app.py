from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)

# Upload folder
UPLOAD_FOLDER = "static/uploads"

# Create uploads folder automatically if not exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# Load trained model
model = tf.keras.models.load_model("model/deep_high_accuracy.h5")

print("Model Loaded Successfully")
print("Model Input Shape:", model.input_shape)


# Image preprocessing
def preprocess_image(path):

    img = cv2.imread(path)

    if img is None:
        raise ValueError("Image not loaded properly")

    # Convert BGR → RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize based on model input
    img = cv2.resize(img, (150,150))   # change to 224 if your model requires

    img = img / 255.0

    img = np.expand_dims(img, axis=0)

    return img


# Home page
@app.route("/")
def home():
    return render_template("index.html")


# Prediction API
@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({"error":"No image uploaded"})

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error":"No selected file"})

    # Save uploaded file
    path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(path)

    # Preprocess image
    img = preprocess_image(path)

    # Model prediction
    prediction = model.predict(img)[0][0]

    print("Raw prediction value:", prediction)

    # Adjust label if necessary
    if prediction > 0.5:
        result = "Real"
    else:
        result = "Fake"

    confidence = round(float(prediction) * 100, 2)

    return jsonify({
        "result": result,
        "confidence": confidence
    })


if __name__ == "__main__":
    app.run(debug=True)