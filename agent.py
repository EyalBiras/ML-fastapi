import torch
from torch import nn
from torchvision import transforms

classes = {
    0: "dog",
    1: "horse",
    2: "elefent",
    3: "butterfly",
    4: "chicken",
    5: "cat",
    6: "cow",
    7: "sheep",
    8: "spider",
    9: "squirrel"
}

FILE = "model.pth"
from neural_network import MyModel

data_dir = r"C:\Users\User\Desktop\עמוס\PycharmProjects\ML fastapi\data\raw-img"
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])




def accurecy_calculation(predictions: torch.tensor, real: torch.tensor) -> int:
    accurecy = torch.eq(predictions, real).sum().item()
    return accurecy / len(real) * 100


class Agent:
    def __init__(self, path_to_model):
        self.PATH_TO_MODEL = path_to_model
        self.model = MyModel(input_channels=3, output_channels=10, hidden_units=8)
        self.model.load_state_dict(torch.load(self.PATH_TO_MODEL))
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=0.001)

    def predict(self, x: torch.tensor) -> str:
        print(torch.argmax(torch.softmax(self.model(x), dim=1)).item())
        print(classes[0])
        return classes[torch.argmax(torch.softmax(self.model(x), dim=1)).item()]

