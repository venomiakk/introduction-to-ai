import numpy as np

from activations import relu, relu_derivative, softmax


class DenseLayer:
    """Fully connected neural network layer storing all neurons in matrix form.

    Each column j of weight matrix W and entry j of bias vector b represent
    the parameters of the j-th neuron in the layer.
    """

    def __init__(self, input_size, output_size, activation="relu", rng=None):
        if activation not in ("relu", "softmax"):
            raise ValueError(f"Unsupported activation '{activation}'. Use 'relu' or 'softmax'.")

        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation

        rng = rng if rng is not None else np.random.default_rng()
        # He initialization for ReLU layers, Xavier initialization for Softmax output layer
        scale = np.sqrt(2.0 / input_size) if activation == "relu" else np.sqrt(1.0 / input_size)
        self.W = rng.normal(0.0, scale, size=(input_size, output_size))
        self.b = np.zeros((1, output_size), dtype=np.float64)

        # Cached values from the most recent forward pass (used in backpropagation)
        self.X = None
        self.Z = None
        self.A = None

        # Gradients computed during the most recent backward pass
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def _activate(self, Z):
        if self.activation == "relu":
            return relu(Z)
        return softmax(Z)

    def forward(self, X, cache=True):
        """Computes layer output A = activation(X @ W + b)."""
        Z = X @ self.W + self.b
        A = self._activate(Z)
        if cache:
            self.X = X
            self.Z = Z
            self.A = A
        return A

    def backward(self, d_out, is_output_delta=False):
        """Computes gradients dW, db and returns gradient dX for the previous layer.

        Args:
            d_out: Gradient flowing from the next layer (dA for hidden ReLU layers,
                   or pre-computed dZ = (Y_hat - Y) / N for the Softmax output layer).
            is_output_delta: True when d_out is already dZ (Softmax + Cross-Entropy).
        """
        if is_output_delta:
            dZ = d_out
        else:
            dZ = d_out * relu_derivative(self.Z)

        self.dW = self.X.T @ dZ
        self.db = np.sum(dZ, axis=0, keepdims=True)
        dX = dZ @ self.W.T
        return dX

    def update(self, learning_rate):
        """Vanilla gradient descent update (no optimizer, no regularization)."""
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db
