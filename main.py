import os
import shutil
import uuid

import uvicorn
from PIL import Image
from fastapi import FastAPI, File, UploadFile

from agent import Agent, transform

app = FastAPI()

agent = Agent("model.pth")


def preprocess_image(image_file):
    image = Image.open(image_file)

    if image.mode == 'RGBA':
        image = image.convert('RGB')

    image = transform(image).unsqueeze(0)
    return image


@app.post('/model/predict')
def predict_animal(image: UploadFile = File(...)):
    image.filename = f"{uuid.uuid4()}.jpg"

    temp_directory = "temp_images"
    os.makedirs(temp_directory, exist_ok=True)
    image_path = os.path.join(temp_directory, image.filename)
    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    input_image = preprocess_image(image_path)
    print(input_image.shape)

    print(agent.predict(input_image))

    os.remove(image_path)

    return {"predicted_class": agent.predict(input_image)}


def main():
    uvicorn.run(app)


if __name__ == "__main__":
    main()
