import numpy as np
from config import *

def ricker_wavelet(t, f0):
    """Generate a Ricker wavelet."""
    r = (np.pi * f0 * (t - 1.0 / f0))
    return (1.0 - 2.0 * r**2) * np.exp(-r**2)

def forward_propagate(velocity_model):
    """Propagate the wave through the model."""
    # Initialize pressure fields
    p_prev = np.zeros((nx, nz))
    p_curr = np.zeros((nx, nz))
    p_next = np.zeros((nx, nz))

    # Source term
    source = ricker_wavelet(t, f0)

    # Data array
    data = np.zeros((len(x_receiver), nt))

    for i in range(nt):
        # Update pressure field based on the wave equation
        p_next = (2 * p_curr - p_prev +
                  (velocity_model**2 * dt**2) * laplacian(p_curr))

        # Inject source at the source location for the current time step
        p_next[x_source, z_source] += source[i] * dt**2

        # Update pressure fields for the next time step
        p_prev, p_curr = p_curr, p_next

        # Record data at receivers
        for j, xr in enumerate(x_receiver):
            data[j, i] = p_curr[xr, z_receiver]

    return data

def laplacian(field):
    """Compute the Laplacian of a 2D field."""
    return (np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0) +
            np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1) -
            4 * field) / (dx * dz)
