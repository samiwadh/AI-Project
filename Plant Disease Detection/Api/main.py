from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import cv2
import numpy as np
import logging
from io import BytesIO
from PIL import Image # pillow model is that is used to read image
import tensorflow as tf
from tensorflow import keras

app = FastAPI()

Model = tf.keras.models.load_model("D:/A_WORK_DATA/Plant disease Project/saved models/3")

Class_Names = []
@app.get("/ping")
def read_root():
    return " I am saaamiwadho"

def read_file_as_image(data)-> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image file
        file_contents = await file.read()
        
        # Convert file data to image
        image = np.array(Image.open(BytesIO(file_contents)))
        
        # Perform prediction or processing on the image
        # Example: Get the shape of the image
        image_shape = image.shape
        
        # Return prediction results
        return {"image_shape": image_shape}
    
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return {"error": "An internal server error occurred."}


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8002)
