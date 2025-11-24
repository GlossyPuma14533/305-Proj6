import numpy as np
import matplotlib.pyplot as plt
import math

# ------------------------------------------------------------
# Part 3 – Holistic Performance Model and Taylor Series
# ------------------------------------------------------------
# dP/dt = a*C - b/P - c*L
#
# P(t): System performance (normalized throughput)
# C: CPU speed factor (GHz or normalized)
# L: Workload intensity (normalized)
# a, b, c: performance interaction constants
# ------------------------------------------------------------

# Parameters
a = 1.2
b = 3.0
c = 0.4
C = 2.5   # CPU speed (GHz, normalized)
L = 0.3   # Workload intensity
P0 = 10.0 # Initial performance

# ------------------------------------------------------------
# Derivatives for Taylor Expansion
# ------------------------------------------------------------
def compute_derivatives(P0, a, b, c, C, L):
    P1 = a*C - b/P0 - c*L
    P2 = b * P1 / P0**2
    P3 = b * P2 / P0**2 - 2.0 * b * P1**2 / P0**3
    P4 = b * P3 / P0**2 - 6.0 * b * P1 * P2 / P0**3 + 6.0 * b * P1**3 / P0**4
    P5 = b * (
        P0**3 * P4
        - 2.0 * P0**2 * (4.0 * P1 * P3 + 3.0 * P2**2)
        + 36.0 * P0 * P1**2 * P2
        - 24.0 * P1**4
    ) / P0**5
    return [P0, P1, P2, P3, P4, P5]

P_derivs = compute_derivatives(P0, a, b, c, C, L)

# ------------------------------------------------------------
# Taylor Polynomial
# ------------------------------------------------------------
def taylor_poly(t, derivs, order):
    y = np.zeros_like(t, dtype=float)
    for n in range(order + 1):
        y += derivs[n] * t**n / math.factorial(n)
    return y

# ------------------------------------------------------------
# Numerical RK4 Solver
# ------------------------------------------------------------
def f(t, P):
    return a*C - b/P - c*L

def rk4_step(t, P, h):
    k1 = f(t, P)
    k2 = f(t + 0.5*h, P + 0.5*h*k1)
    k3 = f(t + 0.5*h, P + 0.5*h*k2)
    k4 = f(t + h, P + h*k3)
    return P + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def solve_ode_rk4(t0, t_end, h, P0):
    ts, Ps = [t0], [P0]
    t, P = t0, P0
    while t < t_end - 1e-12:
        P = rk4_step(t, P, h)
        t += h
        ts.append(t)
        Ps.append(P)
    return np.array(ts), np.array(Ps)

# ------------------------------------------------------------
# Main Program
# ------------------------------------------------------------
if __name__ == "__main__":
    t0, t_end, h = 0.0, 5.0, 0.001
    t_num, P_num = solve_ode_rk4(t0, t_end, h, P0)

    # -------------------------------
    # Visualization of Taylor series
    # -------------------------------
    plt.figure()
    plt.plot(t_num, P_num, label="Numerical ODE Solution (RK4)", linewidth=2)

    for N in [1, 3, 5]:
        P_T = taylor_poly(t_num, P_derivs, N)
        plt.plot(t_num, P_T, "--", label=f"Taylor Series Order {N}")

    plt.xlabel("Time t (seconds)")
    plt.ylabel("System Performance P(t) (normalized throughput)")
    plt.title("Computer Performance Model\nRK4 vs. Taylor Series Convergence")
    plt.grid(True)
    plt.legend()

    # -------------------------------
    # Error plot
    # -------------------------------
    P_T5 = taylor_poly(t_num, P_derivs, 5)
    error = np.abs(P_num - P_T5)

    plt.figure()
    plt.plot(t_num, error)
    plt.xlabel("Time t (seconds)")
    plt.ylabel("Absolute Error")
    plt.title("Error of 5th-Order Taylor Series Approximation")
    plt.grid(True)

    # -------------------------------
    # Print performance at specific times
    # -------------------------------
    print("\nComputed system performance at selected times:")
    for t_check in [1, 2, 3, 5]:
        idx = (np.abs(t_num - t_check)).argmin()
        print(f"t = {t_check} seconds → P(t) ≈ {P_num[idx]:.4f} (normalized throughput)")

    plt.show()

