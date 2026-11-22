import time
import numpy as np

from evaluation import compute_confusion_matrix
from layers import DenseLayer


class ArtificialNeuralNetwork:

    def __init__(
        self,
        input_size=4,
        hidden_layer_sizes=(16, 12),
        output_size=3,
        learning_rate=0.05,
        epochs=1000,
        seed=None,
    ):
        if len(hidden_layer_sizes) != 2:
            raise ValueError("hidden_layer_sizes must specify exactly 2 hidden layers, e.g. (16, 12).")

        self.input_size = input_size
        self.hidden_layer_sizes = tuple(hidden_layer_sizes)
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.seed = seed

        rng = np.random.default_rng(seed)
        h1, h2 = self.hidden_layer_sizes
        self.layers = [
            DenseLayer(input_size, h1, activation="relu", rng=rng),
            DenseLayer(h1, h2, activation="relu", rng=rng),
            DenseLayer(h2, output_size, activation="softmax", rng=rng),
        ]

    def forward(self, X, cache=True):
        """Runs a forward pass through all layers, returns class probabilities."""
        out = X
        for layer in self.layers:
            out = layer.forward(out, cache=cache)
        return out

    @staticmethod
    def compute_loss(Y_true_onehot, Y_pred):
        """Computes categorical cross-entropy loss averaged over all samples."""
        eps = 1e-15
        Y_pred_clipped = np.clip(Y_pred, eps, 1.0)
        return float(-np.mean(np.sum(Y_true_onehot * np.log(Y_pred_clipped), axis=1)))

    def backward(self, Y_true_onehot, Y_pred):
        """Backpropagates categorical cross-entropy loss through all layers."""
        n_samples = Y_true_onehot.shape[0]
        # Combined derivative of Categorical Cross-Entropy + Softmax with respect to Z_out
        dZ_out = (Y_pred - Y_true_onehot) / n_samples

        grad = self.layers[-1].backward(dZ_out, is_output_delta=True)
        for layer in reversed(self.layers[:-1]):
            grad = layer.backward(grad, is_output_delta=False)

    def update_parameters(self):
        """Applies gradient descent step to all layers."""
        for layer in self.layers:
            layer.update(self.learning_rate)

    def predict_proba(self, X):
        """Returns predicted class probability distributions without mutating layer caches."""
        return self.forward(X, cache=False)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def accuracy(self, X, y_true):
        return float(np.mean(self.predict(X) == y_true))

    def fit(self, X_train, Y_train_onehot, X_test=None, Y_test_onehot=None, verbose=False):
        start_time = time.perf_counter()

        y_train_idx = np.argmax(Y_train_onehot, axis=1)
        y_test_idx = np.argmax(Y_test_onehot, axis=1) if Y_test_onehot is not None else None

        train_loss_history = []
        train_acc_history = []
        test_loss_history = []
        test_acc_history = []

        log_interval = max(1, self.epochs // 10)

        for epoch in range(1, self.epochs + 1):
            # 1. Forward pass on the entire training set
            Y_pred_train = self.forward(X_train, cache=True)

            # 2. Compute training loss and accuracy
            train_loss = self.compute_loss(Y_train_onehot, Y_pred_train)
            train_acc = float(np.mean(np.argmax(Y_pred_train, axis=1) == y_train_idx))

            # 3. Backpropagation & gradient descent update
            self.backward(Y_train_onehot, Y_pred_train)
            self.update_parameters()

            train_loss_history.append(train_loss)
            train_acc_history.append(train_acc)

            # Evaluate on test set after parameter update (if provided)
            if X_test is not None and Y_test_onehot is not None:
                Y_pred_test = self.predict_proba(X_test)
                test_loss = self.compute_loss(Y_test_onehot, Y_pred_test)
                test_acc = float(np.mean(np.argmax(Y_pred_test, axis=1) == y_test_idx))
                test_loss_history.append(test_loss)
                test_acc_history.append(test_acc)

            if verbose and (epoch == 1 or epoch % log_interval == 0 or epoch == self.epochs):
                msg = (
                    f"Epoch {epoch:4d}/{self.epochs} | "
                    f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:6.2f}%"
                )
                if test_loss_history:
                    msg += (
                        f" | Test Loss: {test_loss_history[-1]:.4f} | "
                        f"Test Acc: {test_acc_history[-1] * 100:6.2f}%"
                    )
                print(msg)

        execution_time_ms = (time.perf_counter() - start_time) * 1000.0

        # Final evaluation after the last update
        Y_pred_train_final = self.predict_proba(X_train)
        final_train_loss = self.compute_loss(Y_train_onehot, Y_pred_train_final)
        final_train_acc = float(np.mean(np.argmax(Y_pred_train_final, axis=1) == y_train_idx))

        result = {
            "final_train_loss": final_train_loss,
            "final_train_acc": final_train_acc,
            "train_loss_history": train_loss_history,
            "train_acc_history": train_acc_history,
            "execution_time_ms": execution_time_ms,
        }

        if X_test is not None and Y_test_onehot is not None:
            Y_pred_test_final = self.predict_proba(X_test)
            y_pred_test_idx = np.argmax(Y_pred_test_final, axis=1)
            result["final_test_loss"] = self.compute_loss(Y_test_onehot, Y_pred_test_final)
            result["final_test_acc"] = float(np.mean(y_pred_test_idx == y_test_idx))
            result["test_loss_history"] = test_loss_history
            result["test_acc_history"] = test_acc_history
            result["y_test_pred"] = y_pred_test_idx
            result["y_test_proba"] = Y_pred_test_final
            result["confusion_matrix"] = compute_confusion_matrix(
                y_test_idx, y_pred_test_idx, num_classes=self.output_size
            )

        return result
