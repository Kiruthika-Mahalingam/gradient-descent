import numpy as np
import matplotlib.pyplot as plt
class LinearRegressionOptimizer:
    """this class is used to find best slope m using brute-force search and gradient descent"""
    def __init__(self, x_values, y_values, c_actual=4.0):
        """  Initializing constructor with 3 parameters
        1. x_values-independent variable
        2. y_values-dependent variable
        3. c_actual-intercept(default value 4.0 is assigned) """
        if len(x_values) != len(y_values):
            raise ValueError("x_values and y_values must have the same length")
        """the above 2 lines will ensure that x-actual and y_actual are of same length
#         if not it will show an error"""
        self.x_values = x_values
        self.y_values = y_values
        self.c_actual = c_actual  #Intercept
        self.optimal_m_brute = None  #To store the best slope from brute force
        self.optimal_m_gd = None  #To store the best slope from gradient descent
        self.mse_history = []  #Stores the MSE values for gradient descent
    def brute_force_search(self, m_range=(0, 5), steps=150):
        """Try different values of m and find the one that gives the least error"""
        m_candidates = np.linspace(m_range[0], m_range[1], steps)
        mse_values = []
        for m in m_candidates:
            predictions = m * self.x_values + self.c_actual
            mse = np.mean((self.y_values - predictions) ** 2)
            mse_values.append(mse)
        self.optimal_m_brute = m_candidates[np.argmin(mse_values)]
        # Visualizing error vs slope graph
        plt.figure()
        plt.plot(m_candidates, mse_values, label='MSE vs Slope')
        plt.axvline(self.optimal_m_brute, color='red', linestyle='--', label=f'Best m: {self.optimal_m_brute:.2f}')
        plt.xlabel('Slope (m)')
        plt.ylabel('Mean Squared Error')
        plt.legend()
        plt.title('Finding Best Slope Using Brute Force')
        plt.show()
        return self.optimal_m_brute
    def gradient_descent(self, learning_rate=0.005, max_iterations=1500, convergence_threshold=1e-6):
        """convergence_threshold is used to Stop if the error change is too small"""
        """Use gradient descent to iteratively adjust m until the error is min"""
        m_gd = np.random.uniform(0, 5)  # Starting with a random slope
        mse_history = []
        for _ in range(max_iterations):
            predictions = m_gd * self.x_values + self.c_actual
            gradient = -2 * np.mean(self.x_values * (self.y_values - predictions))
            m_gd -= learning_rate * gradient  # Update m based on gradient
            mse = np.mean((self.y_values - predictions) ** 2)
            mse_history.append(mse)
            if len(mse_history) > 1 and abs(mse_history[-2] - mse_history[-1]) < convergence_threshold:
                break
            """Stop if MSE is no longer decreasing significantly"""
        self.optimal_m_gd = m_gd
        self.mse_history = mse_history
        # Plotting MSE progression
        plt.figure()
        plt.plot(mse_history, label='MSE Reduction Over Time', color='green')
        plt.xlabel('Iterations')
        plt.ylabel('Mean Squared Error')
        plt.legend()
        plt.title('Gradient Descent Optimization')
        plt.show()
        return m_gd
    def plot_results(self):
        """Shows the dataset and how the final regression line fits"""
        if self.optimal_m_gd is None:
            print("Run gradient_descent() first to compute the best-fit line")
            return
        plt.figure()
        plt.scatter(self.x_values, self.y_values, label='Data Points', color='red', alpha=0.6)
        best_fit_line = self.optimal_m_gd * self.x_values + self.c_actual
        plt.plot(self.x_values, best_fit_line, color='red', label=f'Best Fit Line (m={self.optimal_m_gd:.2f})')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.title('Best Fit Line Using Gradient Descent')
        plt.show()
    def visualize_gradient_updates(self):
        """Show how gradient descent updates the regression line"""
        if self.optimal_m_gd is None:
            print("Run gradient_descent() first before plotting updates")
            return
        plt.figure()
        plt.scatter(self.x_values, self.y_values, label='Data Points', color='red', alpha=0.6)
        step_intervals = len(self.mse_history) // 5
        for i in range(0, len(self.mse_history), step_intervals):
            temp_m = self.optimal_m_gd - (self.mse_history[i] / max(self.mse_history)) * 1.5
            temp_line = temp_m * self.x_values + self.c_actual
            plt.plot(self.x_values, temp_line, linestyle='--', alpha=0.5, label=f'Iteration {i}')
        final_line = self.optimal_m_gd * self.x_values + self.c_actual
        plt.plot(self.x_values, final_line, color='red', label='Final Best-Fit Line')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.title('Gradient Descent Learning Process')
        plt.show()
    def compare_methods(self):
        """Comparing LS and GD efficiency"""
        brute_iterations = 150
        gd_iterations = len(self.mse_history)
        efficiency_ratio = brute_iterations / gd_iterations if gd_iterations > 0 else float('inf')
        print(f"Best slope found (Brute-force): {self.optimal_m_brute:.5f}")
        print(f"Best slope found (Gradient Descent): {self.optimal_m_gd:.5f}")
        print(f"GD is approximately {efficiency_ratio:.2f} times faster than LS")






# main execution
if __name__ == "__main__":
    np.random.seed(42)
    num_points = 100
    x_values = np.linspace(-10, 10, num_points)
    true_m = 2.5
    true_c = 4.0
    noise = np.random.normal(0, 5, num_points)
    y_values = true_m * x_values + true_c + noise
    optimizer = LinearRegressionOptimizer(x_values, y_values, true_c)
    optimizer.brute_force_search()
    optimizer.gradient_descent()
    optimizer.plot_results()
    optimizer.visualize_gradient_updates()
    optimizer.compare_methods()
