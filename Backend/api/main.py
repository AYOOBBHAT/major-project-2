from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
import tensorflow as tf
from Class_Names import CLASS_NAMES
from Helper import read_file_as_image
from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy

app = FastAPI()


def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

def top_2_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)

MODEL = tf.keras.models.load_model(
    r"C:\Users\ayoob bhat\Desktop\Plant_Disease_Detection-main 2\Plant_Disease_Detection-main\Backend\api\model.h5",
    custom_objects={'top_2_accuracy': top_2_accuracy, 
                    'top_3_accuracy': top_3_accuracy}
)

@app.get("/ping")
async def ping():
    return {'message': 'Hello, I am alive'}


@app.post("/predict")
async def predict(file: UploadFile = File(...), image_type: str = None):
    image = read_file_as_image(await file.read(),image_type=='GRAYSCALE')
    image_batch = np.expand_dims(image, axis=0)
    
    prediction = MODEL.predict(image_batch)

    class_index = np.argmax(prediction[0])
    print('class_index', class_index)
    predicted_class = CLASS_NAMES[class_index]
    confidence = prediction[0][class_index]

    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, port="8000", host="localhost")
