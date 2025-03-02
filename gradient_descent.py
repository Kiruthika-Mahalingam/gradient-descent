import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
class LinearRegressionOptimizer:
    """
    A simple class to find the best slope (m) for linear regression using 
    brute-force search and gradient descent.
    """
    def __init__(self, x_values: np.ndarray, y_values: np.ndarray, c_actual: float = 4.0):
        """
        Initialize with data and known intercept.
        """
        if len(x_values) != len(y_values):
            raise ValueError("x_values and y_values must have the same length")
        self.x_values = x_values
        self.y_values = y_values
        self.c_actual = c_actual
        self.optimal_m_brute = None
        self.optimal_m_gd = None
        self.mse_history = []
    def plot_data(self):
        """Plots the dataset."""
        plt.scatter(self.x_values, self.y_values, label='Noisy Data', color='purple', alpha=0.6)
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.legend()
        plt.title('Dataset with Noise')
        plt.show()
    def brute_force_search(self, m_range=(0, 5), steps=150):
        """
        Finds the best slope by checking multiple values in a range.
        """
        m_candidates = np.linspace(*m_range, steps)
        mse_values = [np.mean((self.y_values - (m * self.x_values + self.c_actual)) ** 2) for m in m_candidates]
        # Best slope based on lowest error
        self.optimal_m_brute = m_candidates[np.argmin(mse_values)]
        # Plot the error curve
        plt.plot(m_candidates, mse_values, label='MSE vs. Slope (m)')
        plt.axvline(self.optimal_m_brute, color='red', linestyle='--', label=f'Optimal m: {self.optimal_m_brute:.2f}')
        plt.xlabel('Slope (m)')
        plt.ylabel('MSE')
        plt.legend()
        plt.title('Brute-force Search')
        plt.show()
        return self.optimal_m_brute
    def gradient_descent(self, learning_rate=0.005, max_iterations=1500, convergence_threshold=1e-6):
        """
        Uses gradient descent to minimize MSE and find the best slope.
        """
        m_gd = np.random.uniform(0, 5)  # Random start
        mse_history = []
        for _ in range(max_iterations):
            predictions = m_gd * self.x_values + self.c_actual
            gradient = -2 * np.mean(self.x_values * (self.y_values - predictions))
            m_gd -= learning_rate * gradient  # Update slope
            mse = np.mean((self.y_values - predictions) ** 2)
            mse_history.append(mse)
            # Stop if the error is not changing much
            if len(mse_history) > 1 and abs(mse_history[-2] - mse_history[-1]) < convergence_threshold:
                break
        self.optimal_m_gd = m_gd
        self.mse_history = mse_history
        # Plot MSE reduction over iterations
        plt.plot(mse_history, label='Gradient Descent MSE')
        plt.xlabel('Iterations')
        plt.ylabel('MSE')
        plt.title('Gradient Descent Convergence')
        plt.legend()
        plt.show()
        return m_gd
    def compare_efficiency(self):
        """
        Compares brute-force search and gradient descent in terms of efficiency.
        """
        brute_iterations = 150  # Fixed search space
        gd_iterations = len(self.mse_history)

        efficiency_ratio = brute_iterations / gd_iterations if gd_iterations > 0 else float('inf')
        print(f"Optimal slope m (Brute-force): {self.optimal_m_brute:.5f}")
        print(f"Optimal slope m (Gradient Descent): {self.optimal_m_gd:.5f}")
        print(f"Gradient Descent was about {efficiency_ratio:.2f} times more efficient.")

# main execution
if __name__ == "__main__":
    np.random.seed(42)
    # Generating sample data
    points_count = 100
    x_values = np.linspace(-10, 10, points_count)
    m_actual = 2.5
    c_actual = 4.0
    random_noise = np.random.normal(0, 5, points_count)
    y_values = m_actual * x_values + c_actual + random_noise
    # Initialize optimizer
    optimizer = LinearRegressionOptimizer(x_values, y_values, c_actual)
    # Running methods
    optimizer.plot_data()
    optimizer.brute_force_search()
    optimizer.gradient_descent()
    optimizer.compare_efficiency()
