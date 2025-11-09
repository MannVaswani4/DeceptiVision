import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.models.face_cnn import FaceCNN


def fine_tune_facecnn(
    train_dir="../data/deception/train",
    val_dir="../data/deception/val",
    pretrained_path="../models/emotion_cnn.pth",
    save_path="../models/face_finetuned.pth",
    num_epochs=5,
    lr=1e-4
):
    """
    Fine-tunes the pretrained FaceCNN model on a deception (truth/lie) dataset.
    """

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    transform = transforms.Compose([
        transforms.Grayscale(),      # Convert to 1 channel
        transforms.Resize((48, 48)), # Match original input size
        transforms.ToTensor()
    ])

    train_data = datasets.ImageFolder(train_dir, transform=transform)
    val_data = datasets.ImageFolder(val_dir, transform=transform)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

    print("Classes:", train_data.classes)

    model = FaceCNN(num_emotions=7)
    model.load_state_dict(torch.load(pretrained_path, map_location=device))
    print("Pretrained emotion model loaded.")

    # Replace last layer for 2-class output (truth vs lie)
    model.fc[-1] = nn.Linear(model.fc[-1].in_features, 2)

    # Freeze earlier convolutional layers (optional)
    for param in model.conv.parameters():
        param.requires_grad = False

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs} | Training Loss: {avg_loss:.4f}")

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        print(f"Validation Accuracy: {val_acc:.4f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print("✅ Fine-tuned model saved at:", save_path)


if __name__ == "__main__":
    fine_tune_facecnn(
        train_dir="../data/deception/train",
        val_dir="../data/deception/val",
        pretrained_path="../models/emotion_cnn.pth",
        save_path="../models/face_finetuned.pth",
        num_epochs=5,
        lr=1e-4
    )
