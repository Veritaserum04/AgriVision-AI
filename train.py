import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import accuracy_score

from utils.dataset import (
    train_loader,
    val_loader,
    NUM_CLASSES,
    DEVICE,
)

from models.model import get_model


def train():

    model = get_model(NUM_CLASSES)
    model.to(DEVICE)

    best_accuracy = 0.0

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.0001
    )

    EPOCHS = 10

    for epoch in range(EPOCHS):

        # -------------------------
        # TRAINING
        # -------------------------

        model.train()

        train_loss = 0

        for images, labels in train_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # -------------------------
        # VALIDATION
        # -------------------------

        model.eval()

        predictions = []
        actual = []

        val_loss = 0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)

                loss = criterion(outputs, labels)

                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)

                predictions.extend(preds.cpu().numpy())
                actual.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)

        accuracy = accuracy_score(actual, predictions)

        print("-" * 50)
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print(f"Train Loss      : {avg_train_loss:.4f}")
        print(f"Validation Loss : {avg_val_loss:.4f}")
        print(f"Accuracy        : {accuracy * 100:.2f}%")

        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy

            torch.save(
                model.state_dict(),
                "models/best_model.pth"
            )

            print("✅ Best model saved!")

    print("\nTraining Complete!")
    print(f"🏆 Best Validation Accuracy: {best_accuracy * 100:.2f}%")


if __name__ == "__main__":
    train()