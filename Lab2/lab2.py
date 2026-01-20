import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet50
import torch.nn as nn
import time
import torch.optim as optim


def lab2(device):
    print("using device:", device)

    if device.type == "cuda":
        print(torch.cuda.get_device_name(0))

    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )

    dataset = datasets.FakeData(
        size=8000, image_size=(3, 224, 224), num_classes=10, transform=transform
    )

    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Train Model
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)

    # Timing
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    epochs = 2
    start = time.time()

    for epoch in range(epochs):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    end = time.time()
    print(f"{device.type} Training time (seconds): ", end - start)


if __name__ == "__main__":
    print(torch.cuda.is_available())
    device = torch.device("cpu")
    lab2(device)
    device = torch.device("cuda")
    lab2(device)
