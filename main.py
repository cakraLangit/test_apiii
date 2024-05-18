from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from fastapi.middleware.cors import CORSMiddleware
import io
from PIL import Image
import uvicorn
import time  # Import the time module
import json
from datetime import datetime
import os
import glob

import uuid

app = FastAPI()

origins = [
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    "http://127.0.0.1:3000",
    "http://localhost:3000",
    "http://10.3.56.80:3000",  # Add local IP if necessary
    "https://1205-182-1-66-13.ngrok-free.app",  # Add your ngrok URL here
    "192.168.141.15",
    "http://192.168.141.15:3000"

]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
model = load_model('model_48x48_1kernel.h5')


model1 = load_model('model_48x48_1kernel.h5')
model2 = load_model('model_48x48.h5')
model3 = load_model('model_48x48_3kernel.h5')

def load_and_prepare_image(img_file):
    # Load the image in grayscale mode
    img = Image.open(io.BytesIO(img_file)).convert('L')
    img = img.resize((48, 48))

    # Convert to numpy array
    img_array = np.array(img)

    # Normalize the image
    img_array = img_array / 255.0

    # Expand dimensions to include batch size
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

@app.post("/predict1")
async def predict_image_class(file: UploadFile = File(...)):
    # Read image file and perform prediction
    image_data = await file.read()
    img_array = load_and_prepare_image(image_data)
    start_time = time.time()
    prediction = model1.predict(img_array)
    elapsed_time = (time.time() - start_time) * 1000

    # Predict and adjust threshold
    threshold = 0.5
    predicted_class = (prediction > threshold).astype(int)
    await file.close()

    # Generate the current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    apimodel1=1

    # Prepare result to save
    result = {
        "id": str(uuid.uuid4()), 
        "timestamp": timestamp,
        "predicted_class": int(predicted_class[0][0]),
        "raw_prediction": float(prediction[0][0]),
        "prediction_time_ms": elapsed_time,
        "api_model": apimodel1

    }
    
    # Read existing data and append the new result
    try:
        with open("predictions.json", "r") as file:
            data = json.load(file)
            data.append(result)
    except (IOError, ValueError):
        data = [result]

    # Save updated data to the JSON file
    with open("predictions.json", "w") as file:
        json.dump(data, file, indent=4)

    return result


@app.post("/predict2")
async def predict_image_class(file: UploadFile = File(...)):
    # Read image file and perform prediction
    image_data = await file.read()
    img_array = load_and_prepare_image(image_data)
    start_time = time.time()
    prediction = model2.predict(img_array)
    elapsed_time = (time.time() - start_time) * 1000

    # Predict and adjust threshold
    threshold = 0.5
    predicted_class = (prediction > threshold).astype(int)
    await file.close()

    # Generate the current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    apimodel=2

    # Prepare result to save
    result = {
        "id": str(uuid.uuid4()), 
        "timestamp": timestamp,
        "predicted_class": int(predicted_class[0][0]),
        "raw_prediction": float(prediction[0][0]),
        "prediction_time_ms": elapsed_time,
        "api_model": apimodel

    }
    
    # Read existing data and append the new result
    try:
        with open("predictions.json", "r") as file:
            data = json.load(file)
            data.append(result)
    except (IOError, ValueError):
        data = [result]

    # Save updated data to the JSON file
    with open("predictions.json", "w") as file:
        json.dump(data, file, indent=4)

    return result




@app.post("/predict3")
async def predict_image_class(file: UploadFile = File(...)):
    # Read image file and perform prediction
    image_data = await file.read()
    img_array = load_and_prepare_image(image_data)
    start_time = time.time()
    prediction = model3.predict(img_array)
    elapsed_time = (time.time() - start_time) * 1000

    # Predict and adjust threshold
    threshold = 0.5
    predicted_class = (prediction > threshold).astype(int)
    await file.close()

    # Generate the current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    apimodel=3

    # Prepare result to save
    result = {
        "id": str(uuid.uuid4()), 
        "timestamp": timestamp,
        "predicted_class": int(predicted_class[0][0]),
        "raw_prediction": float(prediction[0][0]),
        "prediction_time_ms": elapsed_time,
        "api_model": apimodel

    }
    
    # Read existing data and append the new result
    try:
        with open("predictions.json", "r") as file:
            data = json.load(file)
            data.append(result)
    except (IOError, ValueError):
        data = [result]

    # Save updated data to the JSON file
    with open("predictions.json", "w") as file:
        json.dump(data, file, indent=4)

    return result





# @app.post("predict/model1")

# @app.post("predict/model2")

# @app.post("predict/model3")




@app.get("/results/")
async def get_all_results():
    results = []
    # Assume all JSON files are stored in the current directory
    for file_name in glob.glob("predictions.json"):
        with open(file_name, "r") as file:
            result = json.load(file)
            results.append(result)
    return results

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
