import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

RANDOM_SEED = 41
NUM_SAMPLES = 10000
TRAIN_SIZE = int(0.70 * NUM_SAMPLES)
VALIDATION_SIZE = int(0.15 * NUM_SAMPLES)
TEST_SIZE = NUM_SAMPLES - TRAIN_SIZE - VALIDATION_SIZE
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

        # Record history
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

    # Reset seed for each experiment
    set_seed(RANDOM_SEED)

    # Create model
    input_dim = 3 * 64 * 64  # Flattened image
    model = MLP(input_dim, HIDDEN_DIM, num_layers, dropout_p).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Train
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
        'history': history,
        'test_acc': test_acc,
        'test_loss': test_loss,
        'num_layers': num_layers,
        'dropout': dropout_p
    }


def plot_results(results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot accuracy curves
    for result in results:
        name = result['name']
        history = result['history']
        epochs = range(1, len(history['train_acc']) + 1)

        ax1.plot(epochs, history['train_acc'], linestyle='--', label=f"{name} Train", alpha=0.7)
        ax1.plot(epochs, history['val_acc'], linestyle='-', label=f"{name} Val", linewidth=2)

    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Training vs Validation Accuracy', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Plot loss curves
    for result in results:
        name = result['name']
        history = result['history']
        epochs = range(1, len(history['train_loss']) + 1)

        ax2.plot(epochs, history['train_loss'], linestyle='--', label=f"{name} Train", alpha=0.7)
        ax2.plot(epochs, history['val_loss'], linestyle='-', label=f"{name} Val", linewidth=2)

    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('learning_curves.png', dpi=300, bbox_inches='tight')
    print("\nLearning curves saved to 'learning_curves.png'")
    plt.close()


def print_results_table(results):
    print("\n" + "="*80)
    print("FINAL RESULTS TABLE")
    print("="*80)
    print(f"{'Model':<6} {'Layers':<8} {'Dropout':<10} {'Train Acc':<12} {'Val Acc':<12} {'Test Acc':<12}")
    print("-"*80)

    for result in results:
        print(f"{result['name']:<6} {result['num_layers']:<8} {result['dropout']:<10.1f} "
              f"{result['history']['train_acc'][-1]:<12.2f} "
              f"{result['history']['val_acc'][-1]:<12.2f} "
              f"{result['test_acc']:<12.2f}")

    print("="*80)


if __name__ == "__main__":
    device = check_cuda()
    set_seed(RANDOM_SEED)
    train_loader, val_loader, test_loader = build_dataset()

    print(f"\nDataset Info:")
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Val samples: {len(val_loader.dataset)}")
    print(f"  Test samples: {len(test_loader.dataset)}")
    print(f"  Batch size: {DATALOADER_BATCH_SIZE}")
    print(f"  Training epochs: {TRAINING_EPOCHS}")
    print(f"  Hidden dimension: {HIDDEN_DIM}")
    print(f"  Learning rate: {LEARNING_RATE}")

    # Run all four experiments
    results = []

    # M1: 1 layer, no dropout
    results.append(run_experiment("M1", 1, 0.0, train_loader, val_loader, test_loader, device))

    # M2: 3 layers, no dropout
    results.append(run_experiment("M2", 3, 0.0, train_loader, val_loader, test_loader, device))

    # M3: 5 layers, no dropout
    results.append(run_experiment("M3", 5, 0.0, train_loader, val_loader, test_loader, device))

    # M4: 5 layers, dropout 0.4
    results.append(run_experiment("M4", 5, 0.4, train_loader, val_loader, test_loader, device))

    # Print final results table
    print_results_table(results)

    # Create visualizations
    plot_results(results)