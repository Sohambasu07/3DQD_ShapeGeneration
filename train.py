import torch
import torch.nn as nn
import torch.optim as optim


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create the models
encoder = Encoder3D.to(device)
decoder = Decoder3D.to(device)

# Define the loss function
criterion = nn.MSELoss()

# Define the optimizer
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train(model, dataloader, criterion, optimizer, device):
    model.train()  # Set the model to train mode

    for batch_idx, (data, _) in enumerate(dataloader):
        data = data.view(-1, 784).to(device)  # Reshape and move data to device

        optimizer.zero_grad()  # Clear the gradients

        reconstructed_data = model(data)  # Forward pass
        loss = criterion(reconstructed_data, data)  # Compute the loss

        loss.backward()  # Backpropagation
        optimizer.step()  # Update the weights

        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}/{len(dataloader)} Loss: {loss.item()}")