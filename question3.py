import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

np.random.seed(42)
n_samples = 200

cluster1_x = np.random.normal(30, 10, n_samples // 2)
cluster1_y = np.random.normal(30, 10, n_samples // 2)
cluster1_labels = np.zeros(n_samples // 2)

cluster2_x = np.random.normal(70, 10, n_samples // 2)
cluster2_y = np.random.normal(70, 10, n_samples // 2)
cluster2_labels = np.ones(n_samples // 2)

X = np.column_stack([np.concatenate([cluster1_x, cluster2_x]), np.concatenate([cluster1_y, cluster2_y])])
y = np.concatenate([cluster1_labels, cluster2_labels])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def standardize(X, mean=None, std=None):
    if mean is None:
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
    return (X - mean) / std, mean, std


X_train_scaled, mean, std = standardize(X_train)
X_test_scaled, _, _ = standardize(X_test, mean, std)


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -250, 250)))


def mini_batch_generator(X, y, batch_size):
    indices = np.random.permutation(len(X))
    for start_idx in range(0, len(X), batch_size):
        batch_idx = indices[start_idx : start_idx + batch_size]
        yield X[batch_idx], y[batch_idx]


class LogisticRegression:
    def __init__(self, learning_rate=0.01, lambda_reg=0.1, batch_size=32):
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        self.batch_size = batch_size
        self.weights = None
        self.bias = None
        self.training_loss = []

    def fit(self, X, y, epochs=100, verbose=True):
        n_features = X.shape[1]
        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(epochs):
            epoch_loss = 0
            n_batches = 0

            for batch_X, batch_y in mini_batch_generator(X, y, self.batch_size):

                z = np.dot(batch_X, self.weights) + self.bias
                y_pred = sigmoid(z)

                m = len(batch_X)
                dz = y_pred - batch_y
                dw = (1 / m) * np.dot(batch_X.T, dz) + (self.lambda_reg * self.weights)
                db = np.mean(dz)

                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

                batch_loss = self._compute_loss(batch_y, y_pred)
                epoch_loss += batch_loss
                n_batches += 1

            avg_epoch_loss = epoch_loss / n_batches
            self.training_loss.append(avg_epoch_loss)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_epoch_loss:.4f}")

    def predict_proba(self, X):
        z = np.dot(X, self.weights) + self.bias
        return sigmoid(z)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def _compute_loss(self, y_true, y_pred):
        m = len(y_true)
        epsilon = 1e-15
        cross_entropy = -np.mean(y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon))
        l2_reg = (self.lambda_reg / 2) * np.sum(self.weights**2)
        return cross_entropy + l2_reg


model = LogisticRegression(learning_rate=0.1, lambda_reg=0.01, batch_size=32)
model.fit(X_train_scaled, y_train, epochs=100)

y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

train_metrics = {
    "accuracy": accuracy_score(y_train, y_train_pred),
    "precision": precision_score(y_train, y_train_pred),
    "recall": recall_score(y_train, y_train_pred),
    "f1": f1_score(y_train, y_train_pred),
}

test_metrics = {"accuracy": accuracy_score(y_test, y_test_pred), "precision": precision_score(y_test, y_test_pred), "recall": recall_score(y_test, y_test_pred), "f1": f1_score(y_test, y_test_pred)}

print("\nTraining Metrics:")
for metric, value in train_metrics.items():
    print(f"{metric.capitalize()}: {value:.4f}")

print("\nTest Metrics:")
for metric, value in test_metrics.items():
    print(f"{metric.capitalize()}: {value:.4f}")

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(model.training_loss)
plt.title("Training Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)

plt.subplot(2, 1, 2)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

X_mesh = np.c_[xx.ravel(), yy.ravel()]
X_mesh_scaled, _, _ = standardize(X_mesh, mean, std)

Z = model.predict(X_mesh_scaled)
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], c="blue", label="Class 0")
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c="red", label="Class 1")
plt.title("Decision Boundary with L2 Regularization")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("question3.png")
plt.close()
