"""
CNN Trainer — ResNet-50 transfer learning for chest X-ray pneumonia detection.
Generates Grad-CAM visualizations for explainability.

Usage:
    python -m train.train_cnn --epochs 15 --batch-size 32
    python -m train.train_cnn --epochs 1 --batch-size 8   # quick test
"""

import argparse
import sys
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import SAVED_MODELS_DIR, CNN_PARAMS, DATA_DIR
from src.preprocessing.image import get_xray_dataloaders, get_val_transforms
from src.utils.metrics import evaluate_model, plot_confusion_matrix, plot_roc_curve
from src.explainability.gradcam import save_gradcam_plot

import cv2


# ──────────────────────────────────────────────
# Model Architecture
# ──────────────────────────────────────────────
def build_resnet50(num_classes: int = 2, pretrained: bool = True):
    """
    Build ResNet-50 with custom classification head for transfer learning.

    - Freeze early layers (layer1, layer2)
    - Fine-tune later layers (layer3, layer4) + custom head
    """
    from torchvision.models import resnet50, ResNet50_Weights

    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = resnet50(weights=weights)

    # Freeze early layers
    for name, param in model.named_parameters():
        if "layer1" in name or "layer2" in name or "conv1" in name or "bn1" in name:
            param.requires_grad = False

    # Replace classification head
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
    )

    return model


# ──────────────────────────────────────────────
# Training Loop
# ──────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch, returns (loss, accuracy)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="  Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    """Validate model, returns (loss, accuracy, all_preds, all_labels, all_probs)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  Validation", leave=False):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, np.array(all_preds), np.array(all_labels), np.array(all_probs)


# ──────────────────────────────────────────────
# Main Training
# ──────────────────────────────────────────────
def train_cnn(
    epochs: int = None,
    batch_size: int = None,
    learning_rate: float = None,
    data_dir: str = None,
    num_workers: int = 4,
):
    """Full CNN training pipeline with Grad-CAM visualizations."""

    epochs = epochs or CNN_PARAMS["epochs"]
    batch_size = batch_size or CNN_PARAMS["batch_size"]
    learning_rate = learning_rate or CNN_PARAMS["learning_rate"]
    data_dir = data_dir or str(DATA_DIR / "chest_xray")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"🫁 Chest X-Ray Pneumonia Detection — ResNet-50")
    print(f"{'='*60}")
    print(f"  Device:      {device}")
    print(f"  Epochs:      {epochs}")
    print(f"  Batch size:  {batch_size}")
    print(f"  LR:          {learning_rate}")
    print(f"  Data dir:    {data_dir}")

    # ─── Data ─────────────────────────────────
    print("\n📊 Loading chest X-ray data...")
    train_loader, val_loader, test_loader = get_xray_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    if len(train_loader.dataset) == 0:
        print("  ⚠ No training images found!")
        print(f"  → Download the dataset: python -m data.download_datasets")
        print(f"  → Or manually place images in: {data_dir}/train/NORMAL/ and {data_dir}/train/PNEUMONIA/")
        return None

    # ─── Model ────────────────────────────────
    print("\n🧠 Building ResNet-50 model...")
    model = build_resnet50(num_classes=CNN_PARAMS["num_classes"])
    model = model.to(device)

    # Count trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {trainable:,} trainable / {total:,} total")

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=CNN_PARAMS["weight_decay"],
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

    # Output directory
    cnn_dir = SAVED_MODELS_DIR / "chest_xray"
    cnn_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = cnn_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # ─── Training Loop ─────────────────────
    best_val_loss = float("inf")
    best_model_wts = None
    patience_counter = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    print(f"\n🚀 Starting training...")
    for epoch in range(epochs):
        print(f"\n  Epoch {epoch+1}/{epochs}")

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc, val_preds, val_labels, val_probs = validate(
            model, val_loader, criterion, device
        )

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"    Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"    Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        print(f"    LR: {current_lr:.6f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print(f"    ✓ Best model saved (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= CNN_PARAMS["patience"]:
                print(f"\n  ⏹ Early stopping at epoch {epoch+1}")
                break

    # ─── Load Best Model ───────────────────
    if best_model_wts:
        model.load_state_dict(best_model_wts)

    # ─── Test Evaluation ───────────────────
    print(f"\n📊 Evaluating on test set...")
    test_loss, test_acc, test_preds, test_labels, test_probs = validate(
        model, test_loader, criterion, device
    )

    test_metrics = evaluate_model(test_labels, test_preds, test_probs)
    print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Test F1:       {test_metrics['f1_score']:.4f}")
    if test_metrics["roc_auc"]:
        print(f"  Test AUC:      {test_metrics['roc_auc']:.4f}")

    # ─── Plots ─────────────────────────────
    # Training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(train_losses, label="Train", color="#2196F3", lw=2)
    axes[0].plot(val_losses, label="Validation", color="#EF5350", lw=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(train_accs, label="Train", color="#2196F3", lw=2)
    axes[1].plot(val_accs, label="Validation", color="#EF5350", lw=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy Curves")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.suptitle("ResNet-50 Training History", fontsize=14, fontweight="bold")
    plt.tight_layout()
    curves_path = str(plots_dir / "training_curves.png")
    fig.savefig(curves_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Confusion matrix
    plot_confusion_matrix(
        test_labels, test_preds,
        labels=["Normal", "Pneumonia"],
        save_path=str(plots_dir / "confusion_matrix.png"),
    )

    # ROC curve
    if test_metrics["roc_auc"]:
        plot_roc_curve(
            test_labels, test_probs,
            save_path=str(plots_dir / "roc_curve.png"),
        )

    # ─── Grad-CAM ──────────────────────────
    print(f"\n🔍 Generating Grad-CAM visualizations...")
    try:
        model.eval()
        # Get a few test images for Grad-CAM
        xray_dir = Path(data_dir)
        test_images_dir = xray_dir / "test"
        gradcam_count = 0

        for cls in ["NORMAL", "PNEUMONIA"]:
            cls_dir = test_images_dir / cls
            if not cls_dir.exists():
                continue
            image_paths = list(cls_dir.glob("*.jpeg")) + list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png"))
            for img_path in image_paths[:3]:  # 3 per class
                # Load original image
                original = cv2.imread(str(img_path))
                if original is None:
                    continue
                original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

                # Preprocess for model
                transform = get_val_transforms()
                augmented = transform(image=original)
                input_tensor = augmented["image"].unsqueeze(0).to(device)

                # Generate Grad-CAM
                save_gradcam_plot(
                    model, input_tensor, original,
                    save_path=str(plots_dir / f"gradcam_{cls.lower()}_{gradcam_count}.png"),
                    target_layer="layer4",
                )
                gradcam_count += 1

        print(f"  ✓ Generated {gradcam_count} Grad-CAM visualizations")
    except Exception as e:
        print(f"  ⚠ Grad-CAM generation failed: {e}")

    # ─── Save Model ────────────────────────
    model_path = cnn_dir / "resnet50_pneumonia.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "num_classes": CNN_PARAMS["num_classes"],
        "image_size": CNN_PARAMS["image_size"],
        "test_metrics": test_metrics,
    }, model_path)

    # Save metrics
    from src.utils.metrics import save_metrics_json
    save_metrics_json(test_metrics, str(cnn_dir / "metrics.json"))

    print(f"\n✅ CNN Training Complete!")
    print(f"   Model:   {model_path}")
    print(f"   Metrics: {cnn_dir / 'metrics.json'}")
    print(f"   Plots:   {plots_dir}")

    return test_metrics


def main():
    parser = argparse.ArgumentParser(description="Train ResNet-50 for Chest X-Ray")
    parser.add_argument("--epochs", type=int, default=CNN_PARAMS["epochs"])
    parser.add_argument("--batch-size", type=int, default=CNN_PARAMS["batch_size"])
    parser.add_argument("--lr", type=float, default=CNN_PARAMS["learning_rate"])
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    train_cnn(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        data_dir=args.data_dir,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
