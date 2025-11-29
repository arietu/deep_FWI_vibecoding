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

    for i in range(n_iterations):
        print(f"Iteration {i+1}/{n_iterations}")

        # Generate synthetic data and wavefield history with the current velocity model
        synthetic_data, wavefield_history = forward_propagate(velocity_current, store_history=True)

        # Calculate misfit
        misfit = calculate_misfit(observed_data, synthetic_data)
        print(f"  Misfit: {misfit}")

        # Calculate gradient
        gradient = calculate_gradient(observed_data, synthetic_data, wavefield_history, velocity_current)

        # Normalize the gradient to prevent extreme updates
        gradient_max = np.abs(gradient).max()
        if gradient_max > 0:
            gradient /= gradient_max

        # Update model
        velocity_current = update_model(velocity_current, gradient, learning_rate)
        print(f"DEBUG: max of velocity_current is {np.max(velocity_current)}")
        print(f"DEBUG: max of velocity_current position {np.where(velocity_current == np.max(velocity_current))}")

    print("\n--- Inversion Finished ---")

    # Plot the results
    # Determine common color bar limits for all plots
    vmin = min(velocity_true.min(), velocity_initial.min(), velocity_current.min())
    vmax = max(velocity_true.max(), velocity_initial.max(), velocity_current.max())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.subplots_adjust(right=0.85) # Adjust the main plot area to make space for the colorbar

    plot_items = [
        (axes[0], velocity_true, "True Velocity"),
        (axes[1], velocity_initial, "Initial Velocity"),
        (axes[2], velocity_current, "Inverted Velocity")
    ]

    for ax, vel, title in plot_items:
        im = ax.imshow(vel.T, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.plot(x_source, z_source, 'r*', markersize=12, label="Source")
        ax.plot(x_receiver, np.full_like(x_receiver, z_receiver), 'bv', markersize=8, label="Receivers")
        ax.set_title(title)
        ax.legend()

    # Create a new axis for the colorbar
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    plt.show()


if __name__ == "__main__":
    run_fwi()
