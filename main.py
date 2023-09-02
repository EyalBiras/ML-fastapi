import os
import shutil
import uuid
from typing import Dict

import uvicorn
from PIL import Image
from fastapi import FastAPI, File, UploadFile

from agent import Agent, transform

app = FastAPI()

PATH_TO_MODEL = "model.pth"
agent = Agent(PATH_TO_MODEL)


def preprocess_image(image_file: str) -> Image:
    image = Image.open(image_file)

    if image.mode == 'RGBA':
        image = image.convert('RGB')

    image = transform(image).unsqueeze(0)
    return image


@app.post('/model/predict', status_code=200)
def predict_animal(image: UploadFile = File(...)) -> Dict[str, str]:
    image.filename = f"{uuid.uuid4()}.jpg"

    temp_directory = "temp_images"
    os.makedirs(temp_directory, exist_ok=True)
    image_path = os.path.join(temp_directory, image.filename)
    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    input_image = preprocess_image(image_path)

    os.remove(image_path)

    return {"predicted_class": agent.predict(input_image)}


def main() -> None:
    uvicorn.run(app)


if __name__ == "__main__":
    main()
