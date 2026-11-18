import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib.pyplot as plt


PLOTS_DIR = Path(__file__).resolve().parent / "plots"


def plot_convergence(result, lower_bound, output_path, title="GA Convergence Curve"):
    """Plots best and average makespan across generations."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    generations = range(len(result["best_history"]))
    ax.plot(generations, result["best_history"], label="Best Makespan", linewidth=2.0)
    ax.plot(
        generations,
        result["avg_history"],
        label="Average Makespan",
        linewidth=1.5,
        linestyle="--",
        alpha=0.85,
    )
    ax.axhline(
        lower_bound,
        color="red",
        linestyle=":",
        linewidth=1.5,
        label=f"Theoretical Lower Bound ({lower_bound})",
    )

    ax.set_title(f"{title} (Final Best = {result['best_makespan']})")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Makespan (Time Units)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_gantt_chart(result, output_path, num_tasks=11):
    """Plots a Gantt chart of the best schedule across all resources."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(16, 7))

    cmap = plt.colormaps["tab20"]
    colors = [cmap(i % 20) for i in range(num_tasks)]
    labeled_tasks = set()

    for op in result["schedule"]:
        tid = op["task_id"]
        label = f"Task {tid + 1}" if (num_tasks <= 15 and tid not in labeled_tasks) else None
        labeled_tasks.add(tid)
        ax.barh(
            y=op["resource"],
            width=op["duration"],
            left=op["start_time"],
            height=0.65,
            color=colors[tid],
            edgecolor="black",
            linewidth=0.4,
            alpha=0.9,
            label=label,
        )

    resources = sorted({op["resource"] for op in result["schedule"]})
    ax.set_yticks(resources)
    ax.set_yticklabels([f"Resource {r}" for r in resources])
    ax.set_xlabel("Time")
    ax.set_title(f"Best Job Schedule Gantt Chart (Makespan = {result['best_makespan']})")
    ax.axvline(
        result["best_makespan"],
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Makespan = {result['best_makespan']}",
    )
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_parameter_comparison(group_title, curves, lower_bound, num_runs, output_path):
    """Plots average convergence curves for different values of a tested parameter."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    for label, mean_curve in curves:
        ax.plot(range(len(mean_curve)), mean_curve, linewidth=1.8, label=label)

    ax.axhline(
        lower_bound,
        color="red",
        linestyle=":",
        linewidth=1.3,
        label=f"Lower Bound ({lower_bound})",
    )
    ax.set_title(f"Effect of {group_title} on Convergence (Averaged over {num_runs} Runs)")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Makespan")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
