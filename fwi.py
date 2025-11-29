import numpy as np
from config import *
from forward_model import forward_propagate

def calculate_misfit(observed_data, synthetic_data):
    """Calculate the L2 misfit."""
    return 0.5 * np.sum((synthetic_data - observed_data)**2)

def calculate_gradient(observed_data, synthetic_data, velocity_model):
    """Calculate the gradient of the misfit function."""
    # This is a simplified gradient calculation.
    # A full implementation would involve the adjoint-state method.
    gradient = np.zeros_like(velocity_model)
    
    # Propagate the residual back in time (adjoint modeling)
    residual = synthetic_data - observed_data
    
    # For simplicity, we'll use a placeholder for the gradient.
    # A real implementation requires a more complex calculation.
    # This is a very basic approximation.
    for i in range(len(x_receiver)):
        for j in range(nt):
            gradient[x_receiver[i], z_receiver] += residual[i, j]

    return gradient

def update_model(velocity_model, gradient, alpha):
    """Update the velocity model."""
    return velocity_model - alpha * gradient
