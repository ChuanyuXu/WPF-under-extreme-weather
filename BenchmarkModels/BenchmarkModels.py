import torch.nn as nn
import torch
## CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Conv2d(in_channels=5, out_channels=64, kernel_size=(1, 4))
        nn.ReLU(),
        self.fc = nn.Linear(64 * 1 * 9, 12)

    def forward(self, x):
        x = self.conv(x)
        x = torch.relu(x)
        x = x.reshape(x.size(0), -1)
        output = self.fc(x).unsqueeze(2)
        return output
model_fore_pre = CNN()
optimizer_fore_pre = torch.optim.Adam(model_fore_pre.parameters(), lr=0.001)

## FC
class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()
        self.fc1 = nn.Linear(12 * 5, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 12 * 1)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = x.view(-1, 12, 1)
        return x
model_fore_pre = FC()
optimizer_fore_pre = torch.optim.Adam(model_fore_pre.parameters(), lr=0.001)

## LSTM
class model_fore(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(model_fore, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out)
        return output

dem_realc=5
model_fore_pre = model_fore(input_size=dem_realc, hidden_size=64)
optimizer_fore_pre = torch.optim.Adam(model_fore_pre.parameters(), lr=0.001)

## LeNet5
import torch.nn.functional as F
class LeNet5_TS(nn.Module):
    def __init__(self):
        super(LeNet5_TS, self).__init__()
        self.conv1 = nn.Conv2d(5, 6, kernel_size=(3,1), padding=(1,0))
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(3,1), padding=(1,0))
        self.fc1 = nn.Linear(16 * 1 * 12, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 12)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=(1,1), stride=(1,1))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=(1,1), stride=(1,1))
        x = x.reshape(-1, 16 * 1 * 12)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.view(-1, 12, 1)
        return x
model_fore_pre = LeNet5_TS()
optimizer_fore_pre = torch.optim.Adam(model_fore_pre.parameters(), lr=0.0002)

## ResNet18
from torchvision.models import resnet18
class ResNet18_TS(nn.Module):
    def __init__(self):
        super(ResNet18_TS, self).__init__()
        self.resnet = resnet18(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(5, 64, kernel_size=(7,1), stride=(2,1), padding=(3,0))
        self.resnet.fc = nn.Linear(512, 12)

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(-1, 12, 1)
        return x
model_fore_pre = ResNet18_TS()
optimizer_fore_pre = torch.optim.Adam(model_fore_pre.parameters(), lr=0.001)

## VGG11
class VGG11_TS(nn.Module):
    def __init__(self):
        super(VGG11_TS, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(5, 64, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1)),
            nn.Conv2d(64, 128, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1)),
            nn.Conv2d(128, 256, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1)),
            nn.Conv2d(256, 512, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1)),
            nn.Conv2d(512, 512, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 12, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 12)
        )
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = x.view(-1, 12, 1)
        return x
## 创建生成器，判别器对象
model_fore_pre = VGG11_TS()
loss_fn_1=nn.MSELoss()
optimizer_fore_pre = torch.optim.Adam(model_fore_pre.parameters(), lr=0.001)

## YOLOv3_TinY
class YOLOv3_Tiny_TS(nn.Module):
    def __init__(self):
        super(YOLOv3_Tiny_TS, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(5, 16, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1)),

            nn.Conv2d(16, 32, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1)),

            nn.Conv2d(32, 64, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1)),

            nn.Conv2d(64, 128, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1)),

            nn.Conv2d(128, 256, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1)),

            nn.Conv2d(256, 512, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=(1, 1)),
            nn.ReLU()
        )
        self.fc = nn.Linear(256 * 1 * 12, 12)

    def forward(self, x):
        x = self.layers(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        x = x.view(-1, 12, 1)
        return x
model_fore_pre = YOLOv3_Tiny_TS()
optimizer_fore_pre = torch.optim.Adam(model_fore_pre.parameters(), lr=0.001)