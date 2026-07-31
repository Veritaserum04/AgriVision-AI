import torch

print("=" * 50)
print("🌱 AgriVision AI")
print("=" * 50)

print(f"PyTorch Version : {torch.__version__}")
print(f"MPS Available   : {torch.backends.mps.is_available()}")
print(f"MPS Built       : {torch.backends.mps.is_built()}")

device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using Device    : {device}")
print("=" * 50)