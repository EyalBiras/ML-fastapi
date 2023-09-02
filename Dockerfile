FROM python:3.8

RUN mkdir WorkingDir && cd WorkingDir && echo "create directory successfully"

COPY ["main.py", "agent.py", "neural_network.py", "model.pth", "./"]

COPY requirements.txt requirements.txt

RUN pip3 install --upgrade pip

RUN pip3 install -r requirements.txt

CMD ["python", "/main.py"]