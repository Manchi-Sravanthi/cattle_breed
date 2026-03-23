from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io
import base64
import os

app = Flask(__name__)

# Load model
model = load_model("cattle_model.h5")
class_labels = ['ayrshire', 'brown_swiss', 'holstein', 'jersey', 'red_dane']

def prepare_image(img):
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Home page route
@app.route("/")
def home():
    return render_template("index.html")

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("file")

    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    img = Image.open(file.stream).convert("RGB")
    img_array = prepare_image(img)

    prediction = model.predict(img_array)
    class_idx = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    # Encode image for preview
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    encoded_image = base64.b64encode(buffer.getvalue()).decode()

    return jsonify({
        "prediction": class_labels[class_idx].upper(),
        "confidence": f"{confidence:.2f}",
        "image": f"data:image/png;base64,{encoded_image}"
    })

# Run app (IMPORTANT for Replit)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))  # Replit dynamic port
    app.run(host="0.0.0.0", port=port)
