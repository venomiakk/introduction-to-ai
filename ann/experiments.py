from pathlib import Path
import numpy as np
from openpyxl import Workbook

from data_loader import load_iris_data
from neural_network import ArtificialNeuralNetwork
from plotter import PLOTS_DIR, plot_parameter_comparison


RESULTS_EXCEL = Path(__file__).resolve().parent / "ann_experiment_results.xlsx"

DEFAULT_PARAMS = {
    "hidden_layer_sizes": (16, 12),
    "learning_rate": 0.05,
    "epochs": 1000,
    "standardize": True,
}


def save_results_to_excel(records, output_file=RESULTS_EXCEL):
    wb = Workbook()
    ws = wb.active
    ws.title = "ANN_Experiments"

    ws.append([
        "Experiment Group",
        "Parameter",
        "Value",
        "Avg Train Acc (%)",
        "Avg Test Acc (%)",
        "Best Test Acc (%)",
        "Worst Test Acc (%)",
        "Avg Train Loss",
        "Avg Test Loss",
        "Avg Execution Time (ms)",
    ])

    for rec in records:
        ws.append([
            rec["group"],
            rec["parameter"],
            rec["value"],
            rec["avg_train_acc"],
            rec["avg_test_acc"],
            rec["best_test_acc"],
            rec["worst_test_acc"],
            rec["avg_train_loss"],
            rec["avg_test_loss"],
            rec["avg_time_ms"],
        ])

    wb.save(output_file)


def run_parameter_experiments(num_runs=10):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    experiment_groups = {
        "Hidden Layer Architecture": (
            "hidden_layer_sizes",
            [(4, 4), (8, 6), (16, 12), (32, 16), (64, 32)],
        ),
        "Learning Rate": (
            "learning_rate",
            [0.005, 0.01, 0.05, 0.1, 0.5],
        ),
        "Number of Epochs": (
            "epochs",
            [100, 300, 500, 1000, 2000],
        ),
        "Feature Standardization": (
            "standardize",
            [True, False],
        ),
    }

    all_records = []

    for group_title, (param_name, values) in experiment_groups.items():
        print(f"\n--- Testing {group_title} ({param_name}) ---")
        loss_curves = []
        acc_curves = []

        for val in values:
            params = DEFAULT_PARAMS.copy()
            params[param_name] = val

            run_results = []
            for run_idx in range(num_runs):
                seed = 100 + run_idx
                data = load_iris_data(
                    test_size=0.2,
                    seed=seed,
                    standardize=params["standardize"],
                )
                model = ArtificialNeuralNetwork(
                    input_size=data["X_train"].shape[1],
                    hidden_layer_sizes=params["hidden_layer_sizes"],
                    output_size=len(data["class_names"]),
                    learning_rate=params["learning_rate"],
                    epochs=params["epochs"],
                    seed=seed,
                )
                res = model.fit(
                    data["X_train"],
                    data["Y_train_onehot"],
                    X_test=data["X_test"],
                    Y_test_onehot=data["Y_test_onehot"],
                    verbose=False,
                )
                run_results.append(res)

            train_accs = [r["final_train_acc"] * 100.0 for r in run_results]
            test_accs = [r["final_test_acc"] * 100.0 for r in run_results]
            train_losses = [r["final_train_loss"] for r in run_results]
            test_losses = [r["final_test_loss"] for r in run_results]
            exec_times = [r["execution_time_ms"] for r in run_results]

            avg_train_acc = float(np.mean(train_accs))
            avg_test_acc = float(np.mean(test_accs))
            best_test_acc = float(np.max(test_accs))
            worst_test_acc = float(np.min(test_accs))
            avg_train_loss = float(np.mean(train_losses))
            avg_test_loss = float(np.mean(test_losses))
            avg_time = float(np.mean(exec_times))

            label_val = str(val)
            print(
                f"  {param_name}={label_val:<10} | "
                f"Train Acc: {avg_train_acc:6.2f}% | "
                f"Test Acc: {avg_test_acc:6.2f}% (Best: {best_test_acc:6.2f}%, Worst: {worst_test_acc:6.2f}%) | "
                f"Test Loss: {avg_test_loss:.4f} | Time: {avg_time:6.2f} ms"
            )

            all_records.append({
                "group": group_title,
                "parameter": param_name,
                "value": label_val,
                "avg_train_acc": round(avg_train_acc, 2),
                "avg_test_acc": round(avg_test_acc, 2),
                "best_test_acc": round(best_test_acc, 2),
                "worst_test_acc": round(worst_test_acc, 2),
                "avg_train_loss": round(avg_train_loss, 4),
                "avg_test_loss": round(avg_test_loss, 4),
                "avg_time_ms": round(avg_time, 2),
            })

            mean_loss_curve = np.mean([r["test_loss_history"] for r in run_results], axis=0)
            mean_acc_curve = np.mean([r["test_acc_history"] for r in run_results], axis=0)

            loss_curves.append((f"{label_val} (final={avg_test_loss:.4f})", mean_loss_curve))
            acc_curves.append((f"{label_val} (final={avg_test_acc:.2f}%)", mean_acc_curve))

        plot_file = PLOTS_DIR / f"param_{param_name}.png"
        plot_parameter_comparison(group_title, loss_curves, acc_curves, num_runs, plot_file)

    save_results_to_excel(all_records, RESULTS_EXCEL)
    return all_records
