import torch

print("CUDA available:", torch.cuda.is_available())
print("PyTorch CUDA version:", torch.version.cuda)

if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    x = torch.randn(1000, 1000, device="cuda")
    y = x @ x
    print("Computation device:", y.device)
else:
    print("CUDA is not available. See troubleshooting section.")