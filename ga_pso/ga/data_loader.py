from pathlib import Path
from openpyxl import load_workbook


DEFAULT_EXCEL_PATH = Path(__file__).resolve().parent.parent / "GA_task.xlsx"


def load_tasks_from_excel(excel_path=DEFAULT_EXCEL_PATH):
    """
    Loads tasks from GA_task.xlsx.
    Returns a list of 11 tasks (rows 3..13), where each task is a list of
    50 (resource, duration) tuples read from left to right.
    """
    wb = load_workbook(excel_path, read_only=True, data_only=True)
    rows = list(wb.active.iter_rows(values_only=True))
    wb.close()

    num_ops = len(rows[1]) // 2

    # --- Interpretation 1: 11 tasks (each row = 1 task with 50 operations) ---
    tasks = []
    for row in rows[2:]:
        task = [
            (int(row[2 * op]), int(row[2 * op + 1]))
            for op in range(num_ops)
            if row[2 * op] is not None and row[2 * op + 1] is not None
        ]
        if task:
            tasks.append(task)

    # --- Interpretation 2: 50 tasks (each column pair = 1 task with 11 operations) ---
    # tasks = [[] for _ in range(num_ops)]
    # for row in rows[2:]:
    #     for t in range(num_ops):
    #         res, duration = row[2 * t], row[2 * t + 1]
    #         if res is not None and duration is not None:
    #             tasks[t].append((int(res), int(duration)))

    return tasks


def get_problem_summary(tasks):
    """Computes basic statistics and theoretical lower bound for the JSSP instance."""
    resource_workloads = {}
    task_durations = []

    for task in tasks:
        task_durations.append(sum(d for _, d in task))
        for res, d in task:
            resource_workloads[res] = resource_workloads.get(res, 0) + d

    max_res_workload = max(resource_workloads.values())
    max_task_duration = max(task_durations)

    return {
        "num_tasks": len(tasks),
        "num_ops_per_task": len(tasks[0]),
        "total_operations": sum(len(t) for t in tasks),
        "num_resources": len(resource_workloads),
        "theoretical_lower_bound": max(max_res_workload, max_task_duration),
        "resource_workloads": dict(sorted(resource_workloads.items())),
    }
