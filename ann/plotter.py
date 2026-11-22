import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib.pyplot as plt
import numpy as np


PLOTS_DIR = Path(__file__).resolve().parent / "plots"


def plot_training_curves(result, output_path=None, title="Artificial Neural Network Training History"):
    epochs = range(1, len(result["train_loss_history"]) + 1)

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(13, 5))

    # Left subplot: Cross-Entropy Loss
    ax_loss.plot(
        epochs,
        result["train_loss_history"],
        label=f"Train Loss (Final = {result['final_train_loss']:.4f})",
        linewidth=2.0,
    )
    if "test_loss_history" in result:
        ax_loss.plot(
            epochs,
            result["test_loss_history"],
            label=f"Test Loss (Final = {result['final_test_loss']:.4f})",
            linewidth=2.0,
            linestyle="--",
        )
    ax_loss.set_title("Cross-Entropy Loss per Epoch")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Categorical Cross-Entropy Loss")
    ax_loss.grid(True, linestyle="--", alpha=0.5)
    ax_loss.legend()

    # Right subplot: Classification Accuracy
    train_acc_pct = [a * 100.0 for a in result["train_acc_history"]]
    ax_acc.plot(
        epochs,
        train_acc_pct,
        label=f"Train Accuracy (Final = {result['final_train_acc'] * 100:.2f}%)",
        linewidth=2.0,
    )
    if "test_acc_history" in result:
        test_acc_pct = [a * 100.0 for a in result["test_acc_history"]]
        ax_acc.plot(
            epochs,
            test_acc_pct,
            label=f"Test Accuracy (Final = {result['final_test_acc'] * 100:.2f}%)",
            linewidth=2.0,
            linestyle="--",
        )
    ax_acc.set_title("Classification Accuracy per Epoch")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_ylim(0, 105)
    ax_acc.grid(True, linestyle="--", alpha=0.5)
    ax_acc.legend(loc="lower right")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def plot_confusion_matrix(cm, class_names, output_path=None, title="Test Set Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(7, 6))

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, label="Sample Count")

    short_names = [name.replace("Iris-", "") for name in class_names]
    tick_marks = np.arange(len(short_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(short_names, rotation=20, ha="right")
    ax.set_yticklabels(short_names)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=13, fontweight="bold")

    accuracy = np.trace(cm) / np.sum(cm) * 100.0 if np.sum(cm) > 0 else 0.0
    ax.set_title(f"{title} (Accuracy = {accuracy:.2f}%)")
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")

    fig.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def plot_parameter_comparison(
    group_title,
    loss_curves,
    acc_curves,
    num_runs,
    output_path=None,
):
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(13, 5))

    for label, curve in loss_curves:
        ax_loss.plot(range(1, len(curve) + 1), curve, linewidth=1.8, label=label)
    ax_loss.set_title("Mean Test Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Categorical Cross-Entropy Loss")
    ax_loss.grid(True, linestyle="--", alpha=0.5)
    ax_loss.legend(fontsize=8.5)

    for label, curve in acc_curves:
        ax_acc.plot(range(1, len(curve) + 1), [v * 100.0 for v in curve], linewidth=1.8, label=label)
    ax_acc.set_title("Mean Test Accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_ylim(0, 105)
    ax_acc.grid(True, linestyle="--", alpha=0.5)
    ax_acc.legend(loc="lower right", fontsize=8.5)

    fig.suptitle(
        f"Effect of {group_title} (Averaged over {num_runs} Runs)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()
