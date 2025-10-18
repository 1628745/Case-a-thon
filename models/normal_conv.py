import torch
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

checkpoint = torch.load("classification_model.pt", weights_only=False)

model = checkpoint["model"]

torch.save({
    "model": model.state_dict()
},
"small_conv_model.pt")

exit()
# get pretrained
model = models.efficientnet_v2_s(weights="DEFAULT")  # small version, pretrained on ImageNet

# replace classifier head for 7 classes
num_features = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(num_features, 7)
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from torchvision import transforms

class TextLabelDataset(Dataset):
    def __init__(self, img_dir, label_file, transform=None):
        self.img_dir = Path(img_dir)
        self.transform = transform

        unique_labels = {}

        # read lines from label file
        self.samples = []
        with open(label_file, "r") as f:
            for line in f:
                name, label = line.strip().split()
                if label in unique_labels:
                    label = unique_labels[label]
                else:
                    unique_labels[label] = len(unique_labels)
                    label = unique_labels[label]
                self.samples.append((name, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        name, label = self.samples[idx]
        img_path = self.img_dir / name
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

# -------------------
# Example usage
# -------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = TextLabelDataset("classification_dataset/images/", "classification_dataset/labels.txt", transform=transform)

train_len = int(0.8*len(dataset))
val_len = int(0.1*len(dataset))
test_len = len(dataset) - train_len - val_len
train_ds, val_ds, test_ds = random_split(dataset, [train_len, val_len, test_len])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32)



import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)

train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

fig, ax = plt.subplots()

# ax.set_xlim(0, 10)
ax.set_ylim(0, 1)

##################
# TRAINING LOOP 1
##################

for param in model.features.parameters():
    param.requires_grad = True

for epoch in range(20):
    model.train()

    train_loss_batch = []
    train_accuracy_batch = []

    batch_size = 30
    myiter = iter(train_loader)
    my_images = [(imgs.to(device), labels.to(device)) for imgs, labels in [next(myiter) for _ in range(batch_size)]]
    for imgs, labels in my_images:
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss_batch.append(loss.item())

        # print(outputs)
        # print(labels)
        accuracy = (torch.sum(outputs.argmax(dim=1) == labels) / len(outputs)).item() 
        train_accuracy_batch.append(accuracy)


    # ... compute train_loss and val_loss ...

    train_losses.append(sum(train_loss_batch) / len(train_loss_batch))
    train_accuracies.append(sum(train_accuracy_batch) / len(train_accuracy_batch))
    train_loss_batch = []
    train_accuracy_batch = []


    model.eval()
    myiter = iter(val_loader)
    with torch.no_grad():
        val_losses_batch = []
        val_accuracies_batch = []
        for _ in range(batch_size):
            imgs, labels = next(myiter)
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)

            val_loss = criterion(outputs, labels)
            val_losses_batch.append(val_loss.item())

            val_accuracy = (outputs.argmax(dim=1) == labels).sum() / len(outputs)
            val_accuracies_batch.append(val_accuracy.item())

        val_losses.append(sum(val_losses_batch) / len(val_losses_batch))
        val_accuracies.append(sum(val_accuracies_batch) / len(val_accuracies_batch))

    print(f"{train_losses[-1]:0.3f}, {val_losses[-1]:0.3f}, {train_accuracies[-1]:0.3f}, {val_accuracies[-1]:0.3f}")
    model.train()

print("PRETRAINING FRONT LAYER DONE")

for idx, block in enumerate(model.features):
    print(f"block {idx}")
    if idx < 2:
        for param in block.parameters():
            param.requires_grad = False


for epoch in range(30):
    model.train()

    train_loss_batch = []
    train_accuracy_batch = []

    batch_size = 20
    myiter = iter(train_loader)
    my_images = [(imgs.to(device), labels.to(device)) for imgs, labels in [next(myiter) for _ in range(batch_size)]]
    for imgs, labels in my_images:
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss_batch.append(loss.item())

        # print(outputs)
        # print(labels)
        accuracy = (torch.sum(outputs.argmax(dim=1) == labels) / len(outputs)).item() 
        train_accuracy_batch.append(accuracy)


    # ... compute train_loss and val_loss ...

    train_losses.append(sum(train_loss_batch) / len(train_loss_batch))
    train_accuracies.append(sum(train_accuracy_batch) / len(train_accuracy_batch))
    train_loss_batch = []
    train_accuracy_batch = []


    model.eval()
    myiter = iter(val_loader)
    with torch.no_grad():
        val_losses_batch = []
        val_accuracies_batch = []
        for _ in range(batch_size):
            imgs, labels = next(myiter)
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)

            val_loss = criterion(outputs, labels)
            val_losses_batch.append(val_loss.item())

            val_accuracy = (outputs.argmax(dim=1) == labels).sum() / len(outputs)
            val_accuracies_batch.append(val_accuracy.item())

        val_losses.append(sum(val_losses_batch) / len(val_losses_batch))
        val_accuracies.append(sum(val_accuracies_batch) / len(val_accuracies_batch))

    print(f"{train_losses[-1]:0.3f}, {val_losses[-1]:0.3f}, {train_accuracies[-1]:0.3f}, {val_accuracies[-1]:0.3f}")
    model.train()


ax.plot(range(len(train_losses)), train_losses, label="train_losses")
ax.plot(range(len(val_losses)), val_losses, label="val_losses")
ax.plot(range(len(train_accuracies)), train_accuracies, label="train_accuracies")
ax.plot(range(len(val_accuracies)), val_accuracies, label="val_accuracies")
ax.legend()

plt.show()

torch.save({
    "model": model, 
    },
    "classification_model.pt")

torch.save({
    "optimizer": optimizer
    },
    "classification_optimizer.pt")