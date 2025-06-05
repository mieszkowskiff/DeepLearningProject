import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import InceptionCustom
import torchsummary
from torch.utils.data import random_split


torch.manual_seed(42)

data_dir = "./dataset"
batch_size = 64
num_epochs = 10
learning_rate = 1e-4
num_classes = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.5] * 3, std = [0.5] * 3),
])

dataset = datasets.ImageFolder(root = data_dir, transform = transform)

val_split = 0.2
total_size = len(dataset)
val_size = int(total_size * val_split)
train_size = total_size - val_size

train_dataset_split, val_dataset_split = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset_split, batch_size = batch_size, shuffle = True, num_workers = 4)
val_loader = DataLoader(val_dataset_split, batch_size = batch_size, shuffle = False, num_workers = 4)

model = InceptionCustom(num_classes).to(device)
torchsummary.summary(model, input_size=(3, 320, 320), device = device.type)

model.freeze_base()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        model.eval()
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total

    print(f"Epoch {epoch+1}: Valid Loss = {loss:.4f}, Valid Acc = {accuracy:.4f}")

    torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")
