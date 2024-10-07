import torch
import torch.optim as optim

from nnunetv2.training.nnUNetTrainer.variants.fflunet.nnUNetTrainer_FFLUNetDynamicShift12M import (
    FFLUNetDynamicWindowShift12M,
)

torch.autograd.set_detect_anomaly(True)

# Initialize model and data
model = FFLUNetDynamicWindowShift12M(4, 3).cuda()
t = torch.rand(1, 4, 128, 128, 128).cuda()

# Define a simple optimizer, e.g., SGD, with a learning rate
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Dummy training loop
num_epochs = 5
for epoch in range(num_epochs):
    # Zero the gradients before the forward pass
    optimizer.zero_grad()

    # Forward pass
    z = model(t)
    print(f"Output shape at epoch {epoch}: {z.shape}")

    # Compute dummy loss (sum of outputs in this case)
    l = z.sum()
    print(f"Loss at epoch {epoch}: {l.item()}")

    # Backward pass
    l.backward()

    # Update parameters
    optimizer.step()

    print(f"Epoch {epoch+1} complete!\n")

print("Training complete.")
