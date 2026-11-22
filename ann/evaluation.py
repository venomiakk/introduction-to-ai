import numpy as np


def compute_confusion_matrix(y_true, y_pred, num_classes=3):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def compute_classification_metrics(cm, class_names):
    metrics = []
    for idx, name in enumerate(class_names):
        tp = int(cm[idx, idx])
        fp = int(np.sum(cm[:, idx]) - tp)
        fn = int(np.sum(cm[idx, :]) - tp)
        support = int(np.sum(cm[idx, :]))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        metrics.append({
            "class_name": name,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        })
    return metrics


def numerical_gradient_check(model, X, Y_onehot, eps=1e-5):
    """Verifies analytical backpropagation gradients against centered finite differences.

    Returns:
        max_rel_error: float maximum relative error across all weights and biases.
    """
    Y_pred = model.forward(X, cache=True)
    model.backward(Y_onehot, Y_pred)

    max_rel_error = 0.0
    for layer in model.layers:
        for param, analytical_grad in ((layer.W, layer.dW), (layer.b, layer.db)):
            it = np.nditer(param, flags=["multi_index"], op_flags=["readwrite"])
            while not it.finished:
                idx = it.multi_index
                original_val = param[idx]

                param[idx] = original_val + eps
                loss_plus = model.compute_loss(Y_onehot, model.predict_proba(X))

                param[idx] = original_val - eps
                loss_minus = model.compute_loss(Y_onehot, model.predict_proba(X))

                param[idx] = original_val

                num_grad = (loss_plus - loss_minus) / (2.0 * eps)
                ana_grad = analytical_grad[idx]

                denom = max(1e-8, abs(num_grad) + abs(ana_grad))
                rel_error = abs(num_grad - ana_grad) / denom
                if rel_error > max_rel_error:
                    max_rel_error = float(rel_error)

                it.iternext()

    return max_rel_error
