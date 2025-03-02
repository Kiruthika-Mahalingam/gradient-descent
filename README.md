# gradient-descent

# Linear Regression Optimizer

This repository contains a Python implementation of a **Linear Regression Optimizer** that finds the best slope (`m`) using two methods:

1. **Brute-force search** (iterating over a range of values)
2. **Gradient Descent** (iteratively optimizing the slope)

## Features

- **Dataset Generation**: Creates a synthetic dataset with noise.
- **Brute-force Search**: Finds the optimal slope (`m`) by checking multiple values in a range and computing Mean Squared Error (MSE).
- **Gradient Descent**: Uses iterative optimization to find the best slope.
- **Visualization**: Plots the dataset, error curves, and convergence over iterations.
- **Efficiency Comparison**: Compares brute-force and gradient descent methods in terms of efficiency.

## Installation

Ensure you have Python installed. Install the required libraries using:

```bash
pip install numpy pandas matplotlib

## Example Output
Optimal slope m (Brute-force): 2.55034
Optimal slope m (Gradient Descent): 2.53441
Gradient Descent was approximately 7.14 times more efficient than brute-force.

## Dependencies
numpy
pandas
matplotlib
```
