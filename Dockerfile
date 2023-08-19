FROM python:3.8

COPY main.py .
COPY agent.py .
COPY neural_network.py .
COPY model.pth .

COPY requirements.txt requirements.txt

RUN pip3 install --upgrade pip
RUN pip3 install torch torchvision torchaudio

RUN pip3 install -r requirements.txt

CMD ["python", "/main.py"]