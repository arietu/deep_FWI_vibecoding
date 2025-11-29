import numpy as np
from config import *

def ricker_wavelet(t, f0):
    """Generate a Ricker wavelet."""
    r = (np.pi * f0 * (t - 1.0 / f0))
    return (1.0 - 2.0 * r**2) * np.exp(-r**2)

def forward_propagate(velocity_model, store_history=False):
    """
    Propagate the wave through the model with a damping boundary layer.
    Optionally stores and returns the full wavefield history.
    """
    # --- TEMPORARY DEBUGGING ---
    #print(f"DEBUG: Inside forward_propagate, max of velocity_model is {np.max(velocity_model)}")

    # -------------------------

    # Initialize pressure fields
    p_prev = np.zeros((nx, nz))
    p_curr = np.zeros((nx, nz))
    p_next = np.zeros((nx, nz))

    # Create the damping field for the PML
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

    # Source term
    source = ricker_wavelet(t, f0)

    # Data and history arrays
    data = np.zeros((len(x_receiver), nt))
    wavefield_history = []
    if store_history:
        wavefield_history = np.zeros((nt, nx, nz))

    for i in range(nt):
        # Update pressure field based on the wave equation
        p_next = (2 * p_curr - p_prev +
                  (velocity_model**2 * dt**2) * laplacian(p_curr))

        # Inject source at the source location for the current time step
        p_next[x_source, z_source] += source[i] * dt**2

        # Apply damping to the new pressure field
        p_next *= (1 - damping_field * dt)

        # Update pressure fields for the next time step
        p_prev, p_curr = p_curr, p_next

        # Record data at receivers
        for j, xr in enumerate(x_receiver):
            data[j, i] = p_curr[xr, z_receiver]
        
        # Store wavefield history if requested
        if store_history:
            wavefield_history[i] = p_curr

    if store_history:
        return data, wavefield_history
    else:
        return data

def laplacian(field):
    """Compute the Laplacian of a 2D field."""
    return (np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0) +
            np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1) -
            4 * field) / (dx * dz)
