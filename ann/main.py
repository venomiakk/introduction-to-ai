from data_loader import load_iris_data
from evaluation import compute_classification_metrics, numerical_gradient_check
from experiments import DEFAULT_PARAMS, RESULTS_EXCEL, run_parameter_experiments
from neural_network import ArtificialNeuralNetwork
from plotter import PLOTS_DIR, plot_confusion_matrix, plot_training_curves


def print_confusion_matrix(cm, class_names):
    short_names = [name.replace("Iris-", "") for name in class_names]
    header = "True \\ Pred".ljust(14) + "".join(name.rjust(12) for name in short_names)
    print(header)
    print("-" * len(header))
    for i, name in enumerate(short_names):
        row = name.ljust(14) + "".join(f"{cm[i, j]:12d}" for j in range(len(short_names)))
        print(row)


def main():
    print("=== Laboratory 3, Exercise 1: Artificial Neural Network (Iris Classification) ===")
    print("Architecture: 4 -> 16 (ReLU) -> 12 (ReLU) -> 3 (Softmax)")
    print("Training: Full-dataset Vanilla Gradient Descent (no batching, no optimizer, no regularization)\n")

    # 1. Load Iris dataset from local CSV
    data = load_iris_data(
        test_size=0.2,
        seed=42,
        standardize=DEFAULT_PARAMS["standardize"],
    )
    print(f"Features: {data['feature_names']}")
    print(f"Classes:  {data['class_names']}")
    print(f"Split:    {len(data['X_train'])} train samples / {len(data['X_test'])} test samples\n")

    # 2. Numerical Gradient Check (verifies analytical backpropagation math)
    print("=== Running Numerical Gradient Check ===")
    grad_check_model = ArtificialNeuralNetwork(
        input_size=data["X_train"].shape[1],
        hidden_layer_sizes=(8, 6),
        output_size=len(data["class_names"]),
        seed=0,
    )
    # Use small non-zero biases for the finite-difference check so pre-activations
    # do not sit exactly on the non-differentiable ReLU kink at z = 0.
    for layer in grad_check_model.layers:
        layer.b.fill(0.05)

    max_rel_error = numerical_gradient_check(
        grad_check_model,
        data["X_train"][:15],
        data["Y_train_onehot"][:15],
    )
    status = "PASSED" if max_rel_error < 1e-6 else "FAILED"
    print(f"Max relative gradient error: {max_rel_error:.3e} [{status}]\n")

    # 3. Train default ArtificialNeuralNetwork
    print("=== Training Default ArtificialNeuralNetwork ===")
    model = ArtificialNeuralNetwork(
        input_size=data["X_train"].shape[1],
        hidden_layer_sizes=DEFAULT_PARAMS["hidden_layer_sizes"],
        output_size=len(data["class_names"]),
        learning_rate=DEFAULT_PARAMS["learning_rate"],
        epochs=DEFAULT_PARAMS["epochs"],
        seed=42,
    )
    result = model.fit(
        data["X_train"],
        data["Y_train_onehot"],
        X_test=data["X_test"],
        Y_test_onehot=data["Y_test_onehot"],
        verbose=True,
    )

    print(f"\nFinal Train Loss: {result['final_train_loss']:.4f} | Train Accuracy: {result['final_train_acc'] * 100:.2f}%")
    print(f"Final Test Loss:  {result['final_test_loss']:.4f} | Test Accuracy:  {result['final_test_acc'] * 100:.2f}%")
    print(f"Execution Time:   {result['execution_time_ms']:.2f} ms")

    # 4. Confusion Matrix & Per-Class Metrics
    print("\n=== Test Set Confusion Matrix ===")
    print_confusion_matrix(result["confusion_matrix"], data["class_names"])

    print("\n=== Per-Class Test Metrics ===")
    metrics = compute_classification_metrics(result["confusion_matrix"], data["class_names"])
    print(f"{'Class':<18} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>9}")
    print("-" * 61)
    for m in metrics:
        print(
            f"{m['class_name']:<18} {m['precision']:10.4f} {m['recall']:10.4f} "
            f"{m['f1']:10.4f} {m['support']:9d}"
        )

    # 5. Sample Predictions with Softmax Probabilities
    print("\n=== Sample Test Predictions (First 10 Samples) ===")
    for i in range(min(10, len(data["X_test_raw"]))):
        raw_vals = ", ".join(f"{v:.1f}" for v in data["X_test_raw"][i])
        true_name = data["class_names"][data["y_test"][i]]
        pred_name = data["class_names"][result["y_test_pred"][i]]
        probs = ", ".join(f"{p:.3f}" for p in result["y_test_proba"][i])
        mark = "OK" if true_name == pred_name else "MISS"
        print(
            f"  #{i + 1:2d} [{raw_vals}] -> Pred: {pred_name:<15} "
            f"| True: {true_name:<15} | Softmax: [{probs}] ({mark})"
        )

    # 6. Save Training & Confusion Matrix Plots
    plot_training_curves(result, PLOTS_DIR / "default_training_curves.png")
    plot_confusion_matrix(
        result["confusion_matrix"],
        data["class_names"],
        PLOTS_DIR / "default_confusion_matrix.png",
    )
    print(f"\nSaved default training curves and confusion matrix plots to {PLOTS_DIR}")

    # 7. Run Parameter Study Experiments
    print("\n=== Running ANN Parameter Study Experiments ===")
    run_parameter_experiments(num_runs=10)
    print(f"\nSaved experiment summary table to {RESULTS_EXCEL}")
    print(f"Saved parameter study plots to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
