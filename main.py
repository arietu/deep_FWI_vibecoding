import numpy as np
import matplotlib.pyplot as plt
from config import *
from forward_model import forward_propagate
from fwi import calculate_misfit, calculate_gradient, update_model

def run_fwi():
    """Run the Full Waveform Inversion."""
    # Generate "observed" data with the true velocity model
    print("Generating observed data...")
    observed_data = forward_propagate(velocity_true)

    # Initial velocity model
    velocity_current = np.copy(velocity_initial)

    # Inversion loop
    n_iterations = 10
    learning_rate = 1e-9  # This will likely need tuning

    for i in range(n_iterations):
        print(f"Iteration {i+1}/{n_iterations}")

        # Generate synthetic data with the current velocity model
        synthetic_data = forward_propagate(velocity_current)

        # Calculate misfit
        misfit = calculate_misfit(observed_data, synthetic_data)
        print(f"  Misfit: {misfit}")

        # Calculate gradient
        gradient = calculate_gradient(observed_data, synthetic_data, velocity_current)

        # Update model
        velocity_current = update_model(velocity_current, gradient, learning_rate)

    # Plot the results
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(velocity_true.T, cmap="viridis")
    plt.title("True Velocity")
    plt.subplot(1, 3, 2)
    plt.imshow(velocity_initial.T, cmap="viridis")
    plt.title("Initial Velocity")
    plt.subplot(1, 3, 3)
    plt.imshow(velocity_current.T, cmap="viridis")
    plt.title("Inverted Velocity")
    plt.show()

if __name__ == "__main__":
    run_fwi()
