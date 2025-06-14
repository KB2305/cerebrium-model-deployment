from fastapi import FastAPI, File, UploadFile
from app.model import Preprocessor, OnnxModel
from io import BytesIO
from PIL import Image
import numpy as np

app = FastAPI()
model = OnnxModel()
preprocessor = Preprocessor()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image file
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")

    # Preprocess
    input_tensor = preprocessor.preprocess(image)

    # Run inference
    output = model.predict(input_tensor)
    predicted_class = int(np.argmax(output))

    return {"predicted_class_id": predicted_class}
