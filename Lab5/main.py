import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
import matplotlib.pyplot as plt
import numpy as np

RANDOM_SEED = 1236
NUM_SAMPLES = 1000
TRAIN_SIZE = int(0.70 * NUM_SAMPLES)
VALIDATION_SIZE = int(0.15 * NUM_SAMPLES)
TEST_SIZE = NUM_SAMPLES - TRAIN_SIZE - VALIDATION_SIZE
DATALOADER_BATCH_SIZE = 32
TRAINING_EPOCHS = 10
HIDDEN_DIM = 256
LEARNING_RATE = 0.0001


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

    dataset = datasets.FakeData(
        size=NUM_SAMPLES,
        image_size=(3, 64, 64),
        num_classes=5,
        transform=transform
    )

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [TRAIN_SIZE, VALIDATION_SIZE, TEST_SIZE],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )

    train_loader = DataLoader(train_dataset, batch_size=DATALOADER_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=DATALOADER_BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=DATALOADER_BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader

def build_dataset_real_data():
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Use CIFAR-10 (first 5 classes)
    train_full = datasets.CIFAR10(root='./data', train=True, 
                                   download=True, transform=transform)
    test_full = datasets.CIFAR10(root='./data', train=False, 
                                  download=True, transform=transform)
    
    # Keep only first 5 classes
    train_idx = [i for i, (_, y) in enumerate(train_full) if y < 5]
    test_idx = [i for i, (_, y) in enumerate(test_full) if y < 5]
    
    train_data = torch.utils.data.Subset(train_full, train_idx)
    test_dataset = torch.utils.data.Subset(test_full, test_idx)
    
    # Split train into train + val (70/15 ratio)
    val_size = int(0.15 / 0.85 * len(train_data))
    train_size = len(train_data) - val_size
    
    train_dataset, val_dataset = random_split(
        train_data, [train_size, val_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )

    train_loader = DataLoader(train_dataset, batch_size=DATALOADER_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=DATALOADER_BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=DATALOADER_BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader, test_loader

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_hidden_layers, dropout_p):
        super().__init__()
        assert num_hidden_layers in {1, 3, 5}, "num_hidden_layers must be 1, 3, or 5"
        assert dropout_p in {0.0, 0.4}, "dropout_p must be 0 or 0.4"
        layers = []

        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        if dropout_p > 0:
            layers.append(nn.Dropout(dropout_p))

        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

            if dropout_p > 0:
                layers.append(nn.Dropout(dropout_p))

        layers.append(nn.Linear(hidden_dim, 5))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def train_model(model, train_loader, val_loader, device, epochs=TRAINING_EPOCHS):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    print("\nStarting training...")
    for epoch in range(epochs):
        # Training
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validation
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    return history


def run_experiment(model_name, num_layers, dropout_p, train_loader, val_loader, test_loader, device):
    print(f"Running {model_name}: {num_layers} layers, dropout={dropout_p}")

    input_dim = 3 * 64 * 64
    model = MLP(input_dim, HIDDEN_DIM, num_layers, dropout_p).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    history = train_model(model, train_loader, val_loader, device)

    # Final test evaluation
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    print(f"\nFinal Results for {model_name}:")
    print(f"  Train Acc: {history['train_acc'][-1]}%")
    print(f"  Val Acc:   {history['val_acc'][-1]}%")
    print(f"  Test Acc:  {test_acc:.2f}%")
    print(f"  Train Loss: {history['train_loss'][-1]}")
    print(f"  Val Loss:   {history['val_loss'][-1]}")
    print(f"  Test Loss:  {test_loss}")

    return {
        'name': model_name,
        'model': model,
        'history': history,
        'test_acc': test_acc,
        'test_loss': test_loss,
        'num_layers': num_layers,
        'dropout': dropout_p
    }


def get_confidence(model, data_loader, device):
    model.eval()
    confidences, predictions, labels_all = [], [], []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            probs = F.softmax(logits, dim=1)

            max_probs, preds = torch.max(probs, dim=1)
            confidences.extend(max_probs.cpu().numpy())
            predictions.extend(preds.cpu().numpy())
            labels_all.extend(labels.numpy())

    return np.array(confidences), np.array(predictions), np.array(labels_all)


def plot_histogram(conf_m2, conf_m4):
    plt.figure(figsize=(8, 5))
    plt.hist(conf_m2, bins=30, alpha=0.6, label="M2 (no dropout)")
    plt.hist(conf_m4, bins=30, alpha=0.6, label="M4 (dropout p=0.4)")
    plt.xlabel("Prediction Confidence")
    plt.ylabel("Number of Samples")
    plt.title("Confidence Distribution (Test Set)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("confidence_histogram.png", dpi=300)
    plt.close()


def mean_confidence_per_class(confidences, labels):
    for c in range(5):
        class_conf = confidences[labels == c]
        print(f"Class {c}: mean confidence = {class_conf.mean():.3f}")

if __name__ == "__main__":
    device = check_cuda()
    set_seed(RANDOM_SEED)
    train_loader, val_loader, test_loader = build_dataset()

    results = []
    # results.append(run_experiment("M1", 1, 0.0, train_loader, val_loader, test_loader, device))
    results.append(run_experiment("M2", 3, 0.0, train_loader, val_loader, test_loader, device))
    # results.append(run_experiment("M3", 5, 0.0, train_loader, val_loader, test_loader, device))
    results.append(run_experiment("M4", 5, 0.4, train_loader, val_loader, test_loader, device))

    m2 = next(r for r in results if r['name'] == "M2")
    m4 = next(r for r in results if r['name'] == "M4")

    conf_m2, preds_m2, labels_m2 = get_confidence(m2['model'], test_loader, device)
    conf_m4, preds_m4, labels_m4 = get_confidence(m4['model'], test_loader, device)

    plot_histogram(conf_m2, conf_m4)

    print(f"\nM2 mean confidence: {conf_m2.mean():.3f}")
    print(f"M4 mean confidence: {conf_m4.mean():.3f}")

    print("\nM2 confidence per class:")
    mean_confidence_per_class(conf_m2, labels_m2)

    print("\nM4 confidence per class:")
    mean_confidence_per_class(conf_m4, labels_m4)