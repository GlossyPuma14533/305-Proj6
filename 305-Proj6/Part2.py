import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Part 2 – (x^2 + 4) y'' + y = x
#
# This script does TWO things:
#   1) Uses the power–series recurrence to compute a_n for y(x) = sum a_n x^n
#      about x0 = 0, up to a chosen order N.
#   2) Solves the ODE numerically with a 4th–order Runge–Kutta (RK4) method
#      for the same initial conditions, and compares the numerical solution
#      with the truncated Taylor series on a plot.
#
# You can change the initial conditions y(0) = a0 and y'(0) = a1 below.
# ------------------------------------------------------------

# ---------- SERIES PART: recurrence for coefficients ----------

def series_coeffs(a0, a1, N):
    """
    Compute coefficients a_n for y(x) = sum_{n=0}^N a_n x^n
    solving (x^2 + 4) y'' + y = x, expanded about x = 0.

    Recurrence (derived in the Word document):
        For n = 0:
            8 a2 + a0 = 0        -> a2 = -a0/8
        For n = 1:
            24 a3 + a1 = 1       -> a3 = (1 - a1)/24
        For n >= 2:
            4 (n+2)(n+1) a_{n+2} + [n(n-1) + 1] a_n = 0
            -> a_{n+2} = -[n(n-1) + 1] * a_n / (4 (n+2)(n+1))
    """
    a = np.zeros(N+1, dtype=float)
    a[0] = a0
    if N >= 1:
        a[1] = a1
    if N >= 2:
        a[2] = -a0 / 8.0
    if N >= 3:
        a[3] = (1.0 - a1) / 24.0
    # General recurrence for n >= 2
    for n in range(2, N-1):
        num = -(n*(n-1) + 1.0) * a[n]
        den = 4.0 * (n+2) * (n+1)
        a[n+2] = num / den
    return a

def taylor_eval(x, coeffs):
    """Evaluate y_T(x) = sum a_n x^n at a scalar or array x."""
    x = np.asarray(x)
    y = np.zeros_like(x, dtype=float)
    # Horner-like evaluation for stability
    for n in reversed(range(len(coeffs))):
        y = y * x + coeffs[n]
    return y

# ---------- NUMERICAL PART: RK4 solver for the ODE ----------

def f(x, Y):
    """
    System form:
        y1 = y
        y2 = y'
        y1' = y2
        y2' = (x - y1) / (x^2 + 4)
    """
    y1, y2 = Y
    dy1 = y2
    dy2 = (x - y1) / (x**2 + 4.0)
    return np.array([dy1, dy2])

def rk4_step(x, Y, h):
    k1 = f(x, Y)
    k2 = f(x + 0.5*h, Y + 0.5*h*k1)
    k3 = f(x + 0.5*h, Y + 0.5*h*k2)
    k4 = f(x + h, Y + h*k3)
    return Y + (h/6.0) * (k1 + 2*k2 + 2*k3 + k4)

def solve_rk4(x0, x_end, h, y0, yp0):
    """
    Integrate from x0 to x_end with step size h
    using RK4 for the system definition above.
    """
    xs = [x0]
    Ys = [np.array([y0, yp0], dtype=float)]
    x = x0
    Y = np.array([y0, yp0], dtype=float)

    # Handle both directions (x_end > x0 or x_end < x0)
    step = h if x_end >= x0 else -abs(h)

    while (step > 0 and x < x_end - 1e-12) or (step < 0 and x > x_end + 1e-12):
        Y = rk4_step(x, Y, step)
        x += step
        xs.append(x)
        Ys.append(Y.copy())

    return np.array(xs), np.array(Ys)

# ------------------------------------------------------------
# MAIN: choose initial conditions and run everything
# ------------------------------------------------------------
if __name__ == "__main__":
    # ----- Initial conditions at x = 0 -----
    # You can change these if your assignment specifies particular values.
    a0 = 0.0   # y(0)
    a1 = 0.0   # y'(0)

    # ----- Taylor series coefficients up to N -----
    N = 10  # order of Taylor expansion (>=5 to include terms up to x^5)
    coeffs = series_coeffs(a0, a1, N)

    print("Taylor-series coefficients a_n (n = 0..N):")
    for n in range(N+1):
        print(f"  a_{n} = {coeffs[n]}")

    print("\nFirst few terms of y(x) ≈ sum a_n x^n:")
    print("  y(x) ≈ ", end="")
    for n in range(6):  # show up to x^5 as required (n <= 5)
        sign = " + " if n > 0 and coeffs[n] >= 0 else " - " if n > 0 else ""
        coef_abs = abs(coeffs[n]) if n > 0 else coeffs[n]
        if n == 0:
            print(f"{coef_abs}", end="")
        elif n == 1:
            print(f"{sign}{coef_abs} x", end="")
        else:
            print(f"{sign}{coef_abs} x^{n}", end="")
    print()

    # ----- Numerical solution via RK4 -----
    x0 = 0.0
    x_end = 5.0
    h = 0.001

    xs, Ys = solve_rk4(x0, x_end, h, a0, a1)
    y_num = Ys[:, 0]

    # ----- Evaluate truncated Taylor polynomial on same grid -----
    # Use only terms up to n=5 (as in manual work), but coeffs computed up to N
    coeffs_trunc = coeffs[:6]
    y_taylor = taylor_eval(xs, coeffs_trunc)

    # ----- Plot comparison -----
    plt.figure()
    plt.plot(xs, y_num, label="Numerical solution (RK4)")
    plt.plot(xs, y_taylor, "--", label="Taylor series (n ≤ 5)")
    plt.xlabel("x")
    plt.ylabel("y(x)")
    plt.title("(x^2 + 4) y'' + y = x  –  Numerical vs. Taylor (about x=0)")
    plt.grid(True)
    plt.legend()

    # ----- Plot absolute error -----
    error = np.abs(y_num - y_taylor)
    plt.figure()
    plt.plot(xs, error)
    plt.xlabel("x")
    plt.ylabel("|y_num(x) - y_Taylor(x)|")
    plt.title("Error of n ≤ 5 Taylor Approximation")
    plt.grid(True)

    plt.show()

