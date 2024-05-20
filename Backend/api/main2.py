import cv2
import pytesseract
from datetime import datetime
import time
from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
import tensorflow as tf
from pymongo import MongoClient 
from Class_Names import CLASS_NAMES
from Helper import read_file_as_image
from dotenv import load_dotenv
import os
import ssl
import certifi
from starlette.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, status

# Path to Tesseract executable (change this if necessary)
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

# Load the cascade classifier for license plate detection
plateCascade = cv2.CascadeClassifier("/Users/mohd.shadab/Downloads/Plant_Disease_Detection-main 3/Plant_Disease_Detection-main/Backend/api/haarcascade_russian_plate_number.xml")
minArea = 500

load_dotenv()

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:3001",
    # Add more allowed origins as needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # You can specify the allowed methods, e.g., ["GET", "POST"]
    allow_headers=["*"],  # You can specify the allowed headers, e.g., ["X-Custom-Header"]
)

ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

mongodb_uri=os.getenv('MONGODB_URI')

client= MongoClient(mongodb_uri, tlsCAFile=certifi.where())



# Access database
db = client["major_project_2"]


# Access collection
collection = db["vehicle_entry_exit_timestamps"]


def get_result(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    numberPlates = plateCascade.detectMultiScale(imgGray, 1.1, 4)

    for (x, y, w, h) in numberPlates:
        area = w * h
        if area > minArea:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img, "NumberPlate", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            imgRoi = img[y:y + h, x:x + w]

            # Perform text extraction using Tesseract OCR
            extracted_text = pytesseract.image_to_string(imgRoi)

            # Get the current date and time
            current_time = datetime.now()

            return extracted_text, current_time

@app.get("/ping")
async def ping():
    return {'message': 'Hello, I am alive'}

@app.post("/predict")
async def predict(file: UploadFile = File(...), image_type: str = None):
    print(file)
    try:
        image = read_file_as_image(await file.read(),image_type=='GRAYSCALE')
        extracted_text, current_time = get_result(image)
        print("Extracted Text:", extracted_text)
        print("Timestamp:", current_time)
        extracted_text = extracted_text.replace("\n", "")
    except:
        raise HTTPException(status_code=400, detail="No text found in the image")
    # collection.delete_many({})
    previous_entries = collection.find({"extracted_text": extracted_text}).sort("timestamp", -1)
    previous_entries = list(previous_entries)
    if(len(previous_entries) == 0):
        collection.insert_one({"timestamp": current_time, "extracted_text": extracted_text, "entered": True})
    else:
        last_entry = previous_entries[0]
        if(last_entry["entered"]):
            collection.insert_one({"timestamp": current_time, "extracted_text": extracted_text, "entered": False})
        else:
            collection.insert_one({"timestamp": current_time, "extracted_text": extracted_text, "entered": True})
    
    return {
        'numberPlate': extracted_text,
        'time': current_time
    }
    
@app.get('/all')
async def getAllData():
    data = collection.find({})
    data = list(data)
    vehicles = {}
    for entry in data:
        if entry["extracted_text"] in vehicles:
            vehicles[entry["extracted_text"]].append(entry['timestamp'])
        else:
            vehicles[entry["extracted_text"]] = [entry['timestamp']]

    values = [sorted(v) for v in vehicles.values()]
    keys = vehicles.keys()
    keys = list(keys)
    res = []
    for index, value in enumerate(values):
        numberplate = keys[index]
        for i in range(len(value)):
            temp = {}
            temp['numberplate'] = numberplate
            temp['entry_timestamp'] = value[i]
            if(i + 1 < len(value)):
                temp['exit_timestamp']  = value[i + 1]
                temp['duration'] =  round(((value[i + 1] - value[i]).total_seconds() / 3600) , 2)
            else: 
                temp['exit_timestamp']  = '-'
                temp['duration'] = '-'
            i +=2
            res.append(temp)        
    return res

if __name__ == "__main__":
    uvicorn.run(app, port=8000, host="0.0.0.0")
