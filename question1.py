import matplotlib.pyplot as plt
import numpy as np

# Muhammed Talha Karagül #


# ----- part b ------
def objective_function(x, y):
    # New objective function with similar complexity but different characteristics
    term1 = 2 * np.sin(np.sqrt(x**2 + y**2)) * np.exp(-0.2 * (x**2 + y**2))
    term2 = 3 * np.exp(-((x - 1) ** 2 + (y + 1) ** 2) / 2)
    term3 = -2 * np.exp(-((x + 2) ** 2 + (y - 2) ** 2) / 3)
    return term1 + term2 + term3


def gradient(x, y):
    h = 1e-7  # Small step for numerical gradient
    dx = (objective_function(x + h, y) - objective_function(x - h, y)) / (2 * h)
    dy = (objective_function(x, y + h) - objective_function(x, y - h)) / (2 * h)
    return dx, dy


def gradient_ascent(points, learning_rate=0.01, momentum=0.9, num_iterations=100):
    paths = []
    for point in points:
        x, y = point
        path = [(x, y)]
        vx, vy = 0, 0  # Initialize velocity components
        for _ in range(num_iterations):
            dx, dy = gradient(x, y)
            # Update velocity with momentum
            vx = momentum * vx + learning_rate * dx
            vy = momentum * vy + learning_rate * dy
            # Update position
            x += vx
            y += vy
            path.append((x, y))
        paths.append(path)
    return paths


def gradient_descent(points, learning_rate=0.01, momentum=0.9, num_iterations=100):
    paths = []
    for point in points:
        x, y = point
        path = [(x, y)]
        vx, vy = 0, 0  # Initialize velocity components
        for _ in range(num_iterations):
            dx, dy = gradient(x, y)
            # Update velocity with momentum (negative for descent)
            vx = momentum * vx - learning_rate * dx
            vy = momentum * vy - learning_rate * dy
            # Update position
            x += vx
            y += vy
            path.append((x, y))
        paths.append(path)
    return paths


# Set random seed for reproducibility
np.random.seed(42)

# Create meshgrid for visualization
x = np.linspace(-4, 4, 100)
y = np.linspace(-4, 4, 100)
X, Y = np.meshgrid(x, y)
Z = objective_function(X, Y)

# Plot the objective function
plt.figure(figsize=(10, 8))
plt.contour(X, Y, Z, levels=30, cmap="viridis")
plt.colorbar(label="Function Value")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Contour Plot of Modified Objective Function")
plt.grid(True)
plt.savefig("question1_function.png")
plt.close()

# Generate random starting points in a spiral pattern
t = np.linspace(0, 2 * np.pi, 5)
r = np.linspace(1, 3, 5)
random_points = [(r_i * np.cos(t_i), r_i * np.sin(t_i)) for r_i, t_i in zip(r, t)]

colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEEAD"]
markers = ["o", "s", "^", "D", "p"]

# Gradient Ascent with momentum
ascent_paths = gradient_ascent(random_points, learning_rate=0.05, momentum=0.9, num_iterations=150)
plt.figure(figsize=(10, 8))
plt.contour(X, Y, Z, 30, cmap="viridis", alpha=0.6)
plt.colorbar(label="Function Value")
plt.title("Gradient Ascent Paths with Momentum")
plt.xlabel("X")
plt.ylabel("Y")

for i, path in enumerate(ascent_paths):
    path_array = np.array(path)

    # Plot path with decreasing alpha for better visualization
    for j in range(len(path_array) - 1):
        alpha = 1 - j / len(path_array)
        plt.plot(path_array[j : j + 2, 0], path_array[j : j + 2, 1], color=colors[i], alpha=alpha, linewidth=1.5)

    # Plot start and end points
    plt.plot(path_array[0, 0], path_array[0, 1], color=colors[i], marker="o", markersize=8, label=f"Start {i+1}")
    plt.plot(path_array[-1, 0], path_array[-1, 1], color=colors[i], marker="*", markersize=10)

plt.legend()
plt.grid(True)
plt.savefig("question1_gradient_ascent.png")
plt.close()

# Gradient Descent with momentum
descent_paths = gradient_descent(random_points, learning_rate=0.05, momentum=0.9, num_iterations=150)
plt.figure(figsize=(10, 8))
plt.contour(X, Y, Z, 30, cmap="viridis", alpha=0.6)
plt.colorbar(label="Function Value")
plt.title("Gradient Descent Paths with Momentum")
plt.xlabel("X")
plt.ylabel("Y")

for i, path in enumerate(descent_paths):
    path_array = np.array(path)

    # Plot path with decreasing alpha for better visualization
    for j in range(len(path_array) - 1):
        alpha = 1 - j / len(path_array)
        plt.plot(path_array[j : j + 2, 0], path_array[j : j + 2, 1], color=colors[i], alpha=alpha, linewidth=1.5)

    # Plot start and end points
    plt.plot(path_array[0, 0], path_array[0, 1], color=colors[i], marker="o", markersize=8, label=f"Start {i+1}")
    plt.plot(path_array[-1, 0], path_array[-1, 1], color=colors[i], marker="*", markersize=10)

plt.legend()
plt.grid(True)
plt.savefig("question1_gradient_descent.png")
plt.close()

plt.show()

plt.close("all")
