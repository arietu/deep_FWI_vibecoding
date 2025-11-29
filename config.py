import numpy as np

# Grid
nx = 201
nz = 201
dx = 10.0
dz = 10.0

# Time
nt = 1000
dt = 0.001
t = np.arange(0, nt * dt, dt)

# Source
f0 = 10.0  # Dominant frequency
x_source = 100
z_source = 10

# Receiver
x_receiver = np.arange(0, nx, 10)
z_receiver = 10

# Velocity model
velocity_true = 1500 * np.ones((nx, nz))
velocity_initial = 1400 * np.ones((nx, nz))
