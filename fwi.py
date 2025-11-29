import numpy as np
from config import *
from forward_model import laplacian

def calculate_misfit(observed_data, synthetic_data):
    """Calculate the L2 misfit."""
    return 0.5 * np.sum((synthetic_data - observed_data)**2)

def calculate_gradient(observed_data, synthetic_data, wavefield_history, velocity_model):
    """
    Calculate the gradient of the misfit function using the adjoint-state method.
    This version uses the numerically stable formulation.
    """
    # Initialize gradient
    gradient = np.zeros_like(velocity_model)

    # Calculate the residual
    residual = synthetic_data - observed_data

    # Initialize adjoint pressure fields
    adj_p_prev = np.zeros((nx, nz))
    adj_p_curr = np.zeros((nx, nz))
    adj_p_next = np.zeros((nx, nz))

    # Create the damping field (same as in forward propagation)
    damping_field = np.zeros((nx, nz))
    for i in range(nx):
        for j in range(nz):
            damp_val = 0
            if i < pml_width:
                damp_val += (pml_damping * ((pml_width - i) / pml_width)**2)
            if i > nx - pml_width - 1:
                damp_val += (pml_damping * ((i - (nx - pml_width - 1)) / pml_width)**2)
            if j < pml_width:
                damp_val += (pml_damping * ((pml_width - j) / pml_width)**2)
            if j > nz - pml_width - 1:
                damp_val += (pml_damping * ((j - (nz - pml_width - 1)) / pml_width)**2)
            damping_field[i, j] = damp_val

    # Adjoint (backward) propagation loop
    for i in range(nt - 1, -1, -1):
        # Time-reversed update of the adjoint field
        adj_p_next = (2 * adj_p_curr - adj_p_prev +
                      (velocity_model**2 * dt**2) * laplacian(adj_p_curr))

        # Inject the residual as the adjoint source at receiver locations
        for j, xr in enumerate(x_receiver):
            adj_p_next[xr, z_receiver] += residual[j, i] * dt**2

        # Apply damping
        adj_p_next *= (1 - damping_field * dt)
        
        # Correlate with forward wavefield to compute gradient
        # This is the stable imaging condition.
        adj_p_tt = (adj_p_next - 2 * adj_p_curr + adj_p_prev) / dt**2
        
        # The (2 / velocity_model**3) term is mathematically correct but numerically
        # unstable, causing the gradient to vanish. We remove it and rely on the
        # normalization in the main loop to handle the scale. This is a form of
        # preconditioning.
        gradient -= wavefield_history[i] * adj_p_tt

        # Update adjoint fields for the next (backward) time step
        adj_p_prev, adj_p_curr = adj_p_curr, adj_p_next

    return gradient

def update_model(velocity_model, gradient, alpha):
    """Update the velocity model."""
    return velocity_model - alpha * gradient
