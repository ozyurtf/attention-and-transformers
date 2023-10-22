import torch
import torch.nn as nn
from torchvision import datasets, transforms
from tqdm import tqdm
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

image_size = 128
patch_size = 16
in_channels = 3
embed_dim = 768
num_heads = 8
mlp_dim = 512
num_layers = 1
num_classes = 100
dropout = 0.01
batch_size = 16

model = VisionTransformer(image_size = image_size,
                          patch_size = patch_size,
                          in_channels = in_channels,
                          embed_dim = embed_dim,
                          num_heads = num_heads,
                          mlp_dim = mlp_dim,
                          num_layers = num_layers,
                          dropout = dropout,
                          batch_size = batch_size,
                          num_classes = num_classes).to(device)

# Load the CIFAR-100 dataset
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.Resize(image_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)


# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

# Train the model
num_epochs = 100
best_val_acc = 0
for epoch in tqdm(range(num_epochs)):
    model.train()
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Validate the model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_acc = 100 * correct / total
    print(f"Epoch: {epoch + 1}, Validation Accuracy: {val_acc:.2f}%")

    # Save the best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")

