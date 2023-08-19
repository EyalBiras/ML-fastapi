import torch
import torchvision
from sklearn.model_selection import StratifiedShuffleSplit
from torch import nn
from torch.utils.data import SubsetRandomSampler
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

dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=transform)

labels = dataset.targets

n_splits = 1
split_ratio = 0.8

stratified_splitter = StratifiedShuffleSplit(n_splits=n_splits, test_size=1 - split_ratio)

train_indices, test_indices = next(stratified_splitter.split(dataset, labels))

train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)

batch_size = 32
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)


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

    def train(self, traininig_loops):
        colors = {'purple': '\033[95m',
                  'blue': '\033[94m',
                  'cyan': '\033[96m',
                  'green': '\033[92m',
                  'yellow': '\033[93m',
                  'red': '\033[91m',
                  'black': '\033[0m',
                  'bold': '\033[1m',
                  'underline': '\033[4m'}
        for training_loop in range(traininig_loops):
            print(f"loop: {training_loop}")
            train_loss = 0
            for b, (x_train, y_train) in enumerate(train_loader):
                self.model.train()
                logits = self.model(x_train)
                loss = self.loss_function(logits, y_train)
                train_loss += loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if b % 100 == 0:
                    precentage = b * 32 / 60000
                    amount = int(precentage * 80)
                    progress = '|' * amount
                    u = '|' * (80 - amount)
                    print(f"[{colors['blue']}{colors['bold']}{progress}{colors['black']}{u}]", end='\r')
            print(f"[{colors['blue']}{colors['bold']}{'|' * 80}{colors['black']}]", end='\r\n')
            print(f"averge loss: {train_loss / len(train_loader):.4f}")
        self.model.eval()
        accurecy = 0
        with torch.inference_mode():
            for b, (x_test, y_test) in enumerate(test_loader):
                logits = self.model(x_test)
                loss = self.loss_function(logits, y_test)
                predictions = torch.softmax(logits, dim=1).argmax(dim=1)
                accurecy += accurecy_calculation(predictions, y_test)
        print(f"final: test loss: {loss}, test accurecy:{accurecy / len(test_loader)}% ")
        torch.save(self.model.state_dict(), self.PATH_TO_MODEL)

    def predict(self, x: torch.tensor) -> str:
        print(torch.argmax(torch.softmax(self.model(x), dim=1)).item())
        print(classes[0])
        return classes[torch.argmax(torch.softmax(self.model(x), dim=1)).item()]

