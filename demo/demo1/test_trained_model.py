import torch
import torchvision
from PIL import Image
from torch import nn

image_path = "./airplane.png"  # 网络下载的图片放置地址
image = Image.open(image_path)

image = image.convert("RGB")  # 将图片转化为RGB三通道图片，有的图片有4个通道（多了个透明度）

transform = torchvision.transforms.Compose(
    [torchvision.transforms.Resize((32, 32)), torchvision.transforms.ToTensor()]
)

image = transform(image)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.model1(x)
        return x


model = torch.load("self_model_20.pth", map_location=torch.device("cpu"))
print(model)

image = torch.reshape(image, (1, 3, 32, 32))
model.eval()
with torch.no_grad():
    output = model(image)

print(output.argmax(1))
