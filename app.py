from flask import Flask, render_template, request, redirect, url_for
from ultralytics import YOLO
import os
import cv2
from PIL import Image
import numpy as np

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["RESULT_FOLDER"] = "static/results"

# Ensure folders exist
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["RESULT_FOLDER"], exist_ok=True)

# Load your custom model
model = YOLO("D:/WAQI-TEST-FILES/YOLOtest/best.pt")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]
        if file:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            # Read image using OpenCV
            img = cv2.imread(filepath)

            # Run YOLO detection
            labels = []

        # Run YOLO detection
        results = model(img)
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            label = f"{model.names[cls]}"
            labels.append(label)  # collect label(s)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        result_path = os.path.join(app.config["RESULT_FOLDER"], "result.jpg")
        cv2.imwrite(result_path, img)

        # Send the labels to HTML
        return render_template("index.html", result_img="results/result.jpg", detections=labels)


    return render_template("index.html")

import base64
import io

@app.route("/webcam", methods=["GET"])
def webcam_page():
    return render_template("webcam.html")  # New webcam page


@app.route("/capture", methods=["POST"])
def capture():
    data = request.get_json()
    image_data = data["image"].split(",")[1]  # Remove data URL prefix
    image_bytes = base64.b64decode(image_data)

    # Convert to OpenCV format
    img_array = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
    img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Run YOLO
    results = model(img)
    labels = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        label = f"{model.names[cls]}"
        labels.append(label)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    result_path = os.path.join(app.config["RESULT_FOLDER"], "result.jpg")
    cv2.imwrite(result_path, img)

    # Re-render the webcam page with result
    return render_template("webcam.html", result_img="results/result.jpg", detections=labels)



if __name__ == "__main__":
    app.run(debug=True)
