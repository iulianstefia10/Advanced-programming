# PW4 - train our CIFAR-10 CNN (same code as the notebook) and export the models to ONNX
# for the /pw4/ page of the site. The site reads them with OpenCV, so it does not need PyTorch.
#
# Run (needs torch and torchvision):  python notebooks/pw4_export_models.py
# It writes into app/:
#   pw4_cifar10_cnn.onnx      our CNN, trained 2 epochs on CIFAR-10
#   pw4_mobilenet_v2.onnx     MobileNetV2 pretrained on ImageNet (torchvision)
#   pw4_imagenet_classes.txt  the 1000 ImageNet class names
#
# Скрипт: навчити нашу CNN (як у ноутбуці) і зберегти обидві моделі у формат ONNX для сайту.
# Після зміни тут — запустити знову, щоб оновити .onnx у app/.

import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_DIR = os.path.join(ROOT, "app")
DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data")   # куди завантажити CIFAR-10


# мережа: 2 згортки (шукають візерунки) + 3 повнозв'язні шари (вирішують клас)
# фото 3×32×32 → conv1 (6 фільтрів 5×5) → 6×28×28 → pool → 6×14×14
#             → conv2 (16 фільтрів 5×5) → 16×10×10 → pool → 16×5×5 = 400 чисел
#             → 120 → 84 → 10 (оцінка для кожного класу)
# параметрів: 456 + 2416 + 48120 + 10164 + 850 = 62 006
class Net(nn.Module):
    # same network as in the PW4 notebook
    def __init__(self):
        super(Net, self).__init__()
        # ЗМІНА: більше фільтрів, напр. Conv2d(3, 32, 5) → тоді Conv2d(32, ...) і fc1 зміниться теж
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)        # зменшує картинку вдвічі
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)          # 10 виходів = 10 класів

    # як дані проходять через мережу; relu = від'ємні числа → 0
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # same as view(-1, 400), but works for the ONNX export
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def train_cifar10_cnn():
    torch.manual_seed(0)    # однакові початкові ваги щоразу
    # фото → тензор 0–1 → нормалізація до −1…1 (сайт робить те саме в cifar10_input)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    trainset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform)    # 50 000 фото
    testset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform)    # 10 000 фото
    # ЗМІНА: batch_size=4 → напр. 64 або 128 (швидше, особливо на GPU; точність може змінитися)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

    net = Net()
    criterion = nn.CrossEntropyLoss()     # помилка для класифікації
    # ЗМІНА: швидкість навчання lr=0.001 (напр. 0.01) або інший оптимізатор: optim.Adam(net.parameters(), lr=0.001)
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    start = time.time()
    # ЗМІНА: кількість епох range(2) → напр. range(5) (вища точність, довше навчання)
    for epoch in range(2):
        for inputs, labels in trainloader:
            optimizer.zero_grad()                     # обнулити градієнти з минулого кроку
            loss = criterion(net(inputs), labels)     # прогноз і помилка
            loss.backward()                           # порахувати градієнти
            optimizer.step()                          # підправити ваги
        print(f"epoch {epoch + 1} done")
    print(f"Training time (CPU): {time.time() - start:.1f} s")

    # точність на тестових фото, яких мережа не бачила
    correct = total = 0
    net.eval()
    with torch.no_grad():                             # без градієнтів: тільки прогноз, швидше
        for images, labels in testloader:
            predicted = net(images).argmax(1)         # клас з найбільшою оцінкою
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy on the 10,000 test images: {100 * correct / total:.1f} %")
    return net


# експорт у ONNX: PyTorch проганяє випадкове фото (dummy) потрібного розміру
# і записує всі операції мережі у файл (tracing)
def export(model, path, size):
    model.eval()
    dummy = torch.randn(1, 3, size, size)
    torch.onnx.export(model, dummy, path, input_names=["input"], output_names=["output"],
                      opset_version=12, dynamo=False)
    print("Saved", path, f"({os.path.getsize(path) / 1e6:.2f} MB)")


if __name__ == "__main__":
    net = train_cifar10_cnn()
    export(net, os.path.join(APP_DIR, "pw4_cifar10_cnn.onnx"), 32)

    # pretrained model: MobileNetV2 trained on ImageNet (1000 classes)
    # ЗМІНА: інша готова модель, напр. torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    weights = torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1
    mobilenet = torchvision.models.mobilenet_v2(weights=weights)
    export(mobilenet, os.path.join(APP_DIR, "pw4_mobilenet_v2.onnx"), 224)
    with open(os.path.join(APP_DIR, "pw4_imagenet_classes.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(weights.meta["categories"]) + "\n")
    print("Saved the ImageNet class names")
