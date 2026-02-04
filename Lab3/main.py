import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
import matplotlib.pyplot as plt
import numpy as np

RANDOM_SEED = 42
NUM_SAMPLES = 10000
TRAIN_SIZE = int(0.70 * NUM_SAMPLES)
VALIDATION_SIZE   = int(0.15 * NUM_SAMPLES)
TEST_SIZE  = NUM_SAMPLES - TRAIN_SIZE - VALIDATION_SIZE
DATALOADER_BATCH_SIZE = 32
TRAINING_EPOCHS = 10
HIDDEN_DIM = 256
LEARNING_RATE = 0.001


def check_cuda():
    print("CUDA available:", torch.cuda.is_available())
    print("PyTorch CUDA version:", torch.version.cuda)
    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))
        x = torch.randn(1000, 1000, device="cuda")
        y = x @ x
        print("Computation device:", y.device)
        device = torch.device("cuda")
        return device
    else:
        print("CUDA is not available.")
        device = torch.device("cpu")
        return device

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_dataset():
    transform = transforms.ToTensor()

    # Dataset
    dataset = datasets.FakeData(
    size=NUM_SAMPLES,
    image_size=(3, 64, 64),
    num_classes=5,
    transform=transform
    )

    # Train, Validation, Test Split
    train_dataset, val_dataset, test_dataset = random_split(
    dataset,
    [TRAIN_SIZE, VALIDATION_SIZE, TEST_SIZE],
    generator=torch.Generator().manual_seed(RANDOM_SEED)
    )

    # Create Loaders
    train_loader = DataLoader(train_dataset, batch_size=DATALOADER_BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=DATALOADER_BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=DATALOADER_BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_hidden_layers, dropout_p):
        super().__init__()
        assert num_hidden_layers in {1, 3, 5}, "num_hidden_layers must be 1, 3, or 5"
        assert dropout_p in {0.0, 0.4}, "dropout_p must be 0 or 0.4"
        layers = []

        # Input layer → first hidden layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        if dropout_p > 0:
            layers.append(nn.Dropout(dropout_p))

        # Additional hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

            if dropout_p > 0:
                layers.append(nn.Dropout(dropout_p))

        # Output layer (logits for 5 classes)
        layers.append(nn.Linear(hidden_dim, 5))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)


if __name__ == "__main__":
    device = check_cuda()
    set_seed(RANDOM_SEED)
    train_dataset, val_dataset, test_dataset = build_dataset()



