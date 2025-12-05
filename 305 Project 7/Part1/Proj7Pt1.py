"""
Lorenz Attractor Interactive Visualization
------------------------------------------

This program:

1. Defines the Lorenz system of ODEs:

       dx/dt = σ (y - x)
       dy/dt = x (ρ - z) - y
       dz/dt = x y - β z

2. Uses a 4th-order Runge–Kutta (RK4) method to numerically solve the system.

3. Plots an animated 3-D trajectory (the Lorenz attractor).

4. Provides sliders so the user can interactively change σ, ρ, and β
   and immediately see the effect on the attractor.

Parameters:
    σ (sigma) – "Prandtl number"
        • Controls how strongly x is pulled toward y (diffusion vs. convection).
        • Affects the rate at which nearby trajectories separate horizontally.

    ρ (rho) – "Rayleigh number"
        • Controls the strength of the driving force / heating.
        • For ρ below a threshold, solutions go to a fixed point.
        • For larger ρ, the system becomes chaotic (butterfly effect).

    β (beta)
        • Geometric/physical parameter related to the aspect ratio of the region.
        • Controls how strongly z is damped.
        • Changing β affects the shape and “thickness” of the wings.

Default “chaotic” parameters:
    σ = 10, ρ = 28, β = 8/3
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)


# -----------------------------
# Lorenz system and integrator
# -----------------------------

def lorenz(state, sigma, rho, beta):
    """
    Compute the time derivatives (dx/dt, dy/dt, dz/dt)
    for the Lorenz system at a given state.

    state = (x, y, z)
    """
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return np.array([dx, dy, dz], dtype=float)


def rk4_step(state, dt, sigma, rho, beta):
    """
    Perform a single 4th-order Runge–Kutta step for the Lorenz system.
    """
    k1 = lorenz(state, sigma, rho, beta)
    k2 = lorenz(state + 0.5 * dt * k1, sigma, rho, beta)
    k3 = lorenz(state + 0.5 * dt * k2, sigma, rho, beta)
    k4 = lorenz(state + dt * k3, sigma, rho, beta)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate_lorenz(initial_state, t_max, dt, sigma, rho, beta):
    """
    Numerically integrate the Lorenz system from t = 0 to t = t_max
    with time step dt, using RK4.

    Returns:
        t: 1D array of times
        traj: 2D array of shape (N, 3) with x, y, z coordinates
    """
    n_steps = int(t_max / dt)
    t = np.linspace(0, t_max, n_steps)
    traj = np.zeros((n_steps, 3))
    state = np.array(initial_state, dtype=float)

    for i in range(n_steps):
        traj[i] = state
        state = rk4_step(state, dt, sigma, rho, beta)

    return t, traj


# -----------------------------
# Parameters and initial data
# -----------------------------

# Default Lorenz parameters (chaotic regime)
DEFAULT_SIGMA = 10.0
DEFAULT_RHO = 28.0
DEFAULT_BETA = 8.0 / 3.0

# Time settings
T_MAX = 40.0     # total simulated time
DT = 0.01        # time step

# Initial condition (x0, y0, z0)
INITIAL_STATE = (1.0, 1.0, 1.0)

# Compute initial trajectory
t, traj = integrate_lorenz(INITIAL_STATE, T_MAX, DT,
                           DEFAULT_SIGMA, DEFAULT_RHO, DEFAULT_BETA)


# -----------------------------
# Matplotlib figure and sliders
# -----------------------------

plt.style.use("default")
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot the trajectory (we'll update this later)
line, = ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], lw=0.7)

ax.set_title("Lorenz Attractor (Interactive)")
ax.set_xlabel("x(t)")
ax.set_ylabel("y(t)")
ax.set_zlabel("z(t)")

# Make room at the bottom for sliders
plt.subplots_adjust(left=0.1, bottom=0.25)

# Slider axes: [left, bottom, width, height]
ax_sigma = plt.axes([0.1, 0.15, 0.8, 0.03])
ax_rho   = plt.axes([0.1, 0.10, 0.8, 0.03])
ax_beta  = plt.axes([0.1, 0.05, 0.8, 0.03])

slider_sigma = Slider(
    ax=ax_sigma,
    label="σ (sigma)",
    valmin=0.1,
    valmax=30.0,
    valinit=DEFAULT_SIGMA,
    valstep=0.1
)

slider_rho = Slider(
    ax=ax_rho,
    label="ρ (rho)",
    valmin=0.1,
    valmax=50.0,
    valinit=DEFAULT_RHO,
    valstep=0.5
)

slider_beta = Slider(
    ax=ax_beta,
    label="β (beta)",
    valmin=0.5,
    valmax=10.0,
    valinit=DEFAULT_BETA,
    valstep=0.1
)


def update_plot(_):
    """
    Callback that recomputes the Lorenz trajectory whenever
    a slider value is changed, and updates the line.
    """
    sigma = slider_sigma.val
    rho = slider_rho.val
    beta = slider_beta.val

    _, new_traj = integrate_lorenz(INITIAL_STATE, T_MAX, DT,
                                   sigma, rho, beta)

    line.set_data(new_traj[:, 0], new_traj[:, 1])
    line.set_3d_properties(new_traj[:, 2])

    # Rescale axes to fit new trajectory nicely
    ax.set_xlim(np.min(new_traj[:, 0]), np.max(new_traj[:, 0]))
    ax.set_ylim(np.min(new_traj[:, 1]), np.max(new_traj[:, 1]))
    ax.set_zlim(np.min(new_traj[:, 2]), np.max(new_traj[:, 2]))

    fig.canvas.draw_idle()


# Connect sliders to callback
slider_sigma.on_changed(update_plot)
slider_rho.on_changed(update_plot)
slider_beta.on_changed(update_plot)


# Optional: Reset button
reset_ax = plt.axes([0.8, 0.90, 0.15, 0.05])
reset_button = Button(reset_ax, "Reset Params")

def reset(event):
    slider_sigma.reset()
    slider_rho.reset()
    slider_beta.reset()

reset_button.on_clicked(reset)


# -----------------------------
# Simple "animated" effect
# -----------------------------
# Rather than a heavy FuncAnimation, we can create a trailing
# effect by updating a shorter segment of the trajectory.

TRAIL_LENGTH = 1000  # number of points to show at once
index = 0

def update_frame(event):
    """
    Timer callback to create a simple animated motion along the curve.
    Uses the *current* parameters (so if the user moves sliders,
    animation uses the most recent trajectory).
    """
    global index

    sigma = slider_sigma.val
    rho = slider_rho.val
    beta = slider_beta.val

    _, new_traj = integrate_lorenz(INITIAL_STATE, T_MAX, DT,
                                   sigma, rho, beta)

    # Wrap index around trajectory length
    index = (index + 10) % len(new_traj)
    start = max(index - TRAIL_LENGTH, 0)

    seg = new_traj[start:index]

    line.set_data(seg[:, 0], seg[:, 1])
    line.set_3d_properties(seg[:, 2])

    fig.canvas.draw_idle()


# Use a timer to call update_frame periodically
timer = fig.canvas.new_timer(interval=30)  # milliseconds
timer.add_callback(update_frame, None)
timer.start()

plt.show()

