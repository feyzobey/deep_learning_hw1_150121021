import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
x = np.linspace(0, 12, 15)
true_function = lambda x: 2 * np.sin(x / 2) + 0.5 * x
y = true_function(x) + np.random.normal(0, 1.5, size=len(x))

scaler_x = StandardScaler()
scaler_y = StandardScaler()
x_scaled = scaler_x.fit_transform(x.reshape(-1, 1)).ravel()
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()


def poly_features(x, degree):

    features = np.vstack([x**i for i in range(degree + 1)]).T

    for i in range(features.shape[1]):
        if i > 0:
            features[:, i] = features[:, i] / (np.max(np.abs(features[:, i])) + 1e-10)
    return features


def soft_threshold(x, lambda_):
    return np.sign(x) * np.maximum(np.abs(x) - lambda_, 0)


def train_with_early_stopping(X_train, y_train, X_val, y_val, learning_rate, epochs, patience=50):
    weights = np.zeros(X_train.shape[1])
    best_weights = weights.copy()
    best_val_error = float("inf")
    patience_counter = 0

    for epoch in range(epochs):

        y_pred = X_train @ weights
        error = y_pred - y_train

        grad = (2 / len(X_train)) * X_train.T @ error
        weights -= learning_rate * grad

        val_pred = X_val @ weights
        val_error = np.mean((val_pred - y_val) ** 2)

        if val_error < best_val_error:
            best_val_error = val_error
            best_weights = weights.copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

        if epoch % 100 == 0:
            learning_rate *= 0.95

    return best_weights, best_val_error


###
degree = 1
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_errors = []

X = poly_features(x_scaled, degree)
best_weights = None
min_cv_error = float("inf")

for train_idx, val_idx in kf.split(X):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y_scaled[train_idx], y_scaled[val_idx]

    weights, val_error = train_with_early_stopping(X_train, y_train, X_val, y_val, learning_rate=0.01, epochs=1000)

    cv_errors.append(val_error)

    if val_error < min_cv_error:
        min_cv_error = val_error
        best_weights = weights.copy()

print(f"Linear Regression CV Error: {np.mean(cv_errors):.4f}")

plt.figure(figsize=(10, 6))
plt.scatter(x, y, color="blue", label="Data Points")

x_smooth = np.linspace(min(x), max(x), 100)
x_smooth_scaled = scaler_x.transform(x_smooth.reshape(-1, 1)).ravel()
X_smooth = poly_features(x_smooth_scaled, degree)
y_smooth_scaled = X_smooth @ best_weights
y_smooth = scaler_y.inverse_transform(y_smooth_scaled.reshape(-1, 1)).ravel()

plt.plot(x_smooth, y_smooth, color="red", label="Linear Model")
plt.plot(x_smooth, true_function(x_smooth), "--", color="green", label="True Function")
plt.xlabel("x")
plt.ylabel("y")
plt.title("First-Order Polynomial Fit with Cross-Validation")
plt.legend()
plt.grid(True)
plt.savefig("question2_first_order.png")
plt.close()

##### Part B - 10th Degree Polynomial with Cross-Validation #####
degree = 10
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_errors = []

X = poly_features(x_scaled, degree)
best_weights = None
min_cv_error = float("inf")

for train_idx, val_idx in kf.split(X):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y_scaled[train_idx], y_scaled[val_idx]

    weights, val_error = train_with_early_stopping(X_train, y_train, X_val, y_val, learning_rate=0.001, epochs=2000)

    cv_errors.append(val_error)

    if val_error < min_cv_error:
        min_cv_error = val_error
        best_weights = weights.copy()

print(f"10th Degree Polynomial CV Error: {np.mean(cv_errors):.4f}")

plt.figure(figsize=(10, 6))
plt.scatter(x, y, color="blue", label="Data Points")

x_smooth = np.linspace(min(x), max(x), 100)
x_smooth_scaled = scaler_x.transform(x_smooth.reshape(-1, 1)).ravel()
X_smooth = poly_features(x_smooth_scaled, degree)
y_smooth_scaled = X_smooth @ best_weights
y_smooth = scaler_y.inverse_transform(y_smooth_scaled.reshape(-1, 1)).ravel()

plt.plot(x_smooth, y_smooth, color="red", label="10th Degree Model")
plt.plot(x_smooth, true_function(x_smooth), "--", color="green", label="True Function")
plt.xlabel("x")
plt.ylabel("y")
plt.title("10th Degree Polynomial Fit with Cross-Validation")
plt.legend()
plt.grid(True)
plt.savefig("question2_tenth_order.png")
plt.close()

##### Part C - 10th Degree Polynomial with Lasso Regularization #####
lambda_lasso = 0.01
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_errors = []

X = poly_features(x_scaled, degree)
best_weights = None
min_cv_error = float("inf")

for train_idx, val_idx in kf.split(X):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y_scaled[train_idx], y_scaled[val_idx]

    weights = np.zeros(X_train.shape[1])
    best_epoch_weights = weights.copy()
    best_val_error = float("inf")
    learning_rate = 0.001
    patience_counter = 0

    for epoch in range(2000):

        y_pred = X_train @ weights
        error = y_pred - y_train

        grad = (2 / len(X_train)) * X_train.T @ error

        for j in range(len(weights)):
            weights[j] = soft_threshold(weights[j] - learning_rate * grad[j], lambda_lasso * learning_rate)

        val_pred = X_val @ weights
        val_error = np.mean((val_pred - y_val) ** 2)

        if val_error < best_val_error:
            best_val_error = val_error
            best_epoch_weights = weights.copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= 50:
            break

        if epoch % 100 == 0:
            learning_rate *= 0.95

    cv_errors.append(best_val_error)

    if best_val_error < min_cv_error:
        min_cv_error = best_val_error
        best_weights = best_epoch_weights.copy()

print(f"10th Degree Polynomial with Lasso CV Error: {np.mean(cv_errors):.4f}")
print("\nLasso Coefficients:")
for i, w in enumerate(best_weights):
    print(f"Degree {i}: {w:.4f}")

plt.figure(figsize=(10, 6))
plt.scatter(x, y, color="blue", label="Data Points")

x_smooth = np.linspace(min(x), max(x), 100)
x_smooth_scaled = scaler_x.transform(x_smooth.reshape(-1, 1)).ravel()
X_smooth = poly_features(x_smooth_scaled, degree)
y_smooth_scaled = X_smooth @ best_weights
y_smooth = scaler_y.inverse_transform(y_smooth_scaled.reshape(-1, 1)).ravel()

plt.plot(x_smooth, y_smooth, color="red", label=f"10th Degree (Lasso, λ={lambda_lasso})")
plt.plot(x_smooth, true_function(x_smooth), "--", color="green", label="True Function")
plt.xlabel("x")
plt.ylabel("y")
plt.title("10th Degree Polynomial with Lasso Regularization")
plt.legend()
plt.grid(True)
plt.savefig("question2_tenth_order_ridge.png")
plt.close()
