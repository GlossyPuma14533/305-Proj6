
import numpy as np
import matplotlib.pyplot as plt

# Differential equation:
#     y'' - (x - 2)*y' + 2*y = 0
# Rewrite as a first-order system:
# Let y1 = y, y2 = y'
# Then:
#   y1' = y2
#   y2' = (x - 2)*y2 - 2*y1

def f(x, Y):
    y1, y2 = Y
    dy1 = y2
    dy2 = (x - 2.0)*y2 - 2.0*y1
    return np.array([dy1, dy2])

def rk4_step(x, Y, h):
    k1 = f(x, Y)
    k2 = f(x + 0.5*h, Y + 0.5*h*k1)
    k3 = f(x + 0.5*h, Y + 0.5*h*k2)
    k4 = f(x + h, Y + h*k3)
    return Y + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def solve_ode_rk4(x0, x_end, h, y0, yp0):
    xs = [x0]
    Ys = [np.array([y0, yp0])]
    x = x0
    Y = np.array([y0, yp0])
    # Integrate forward or backward depending on the sign of h
    step = h if x_end >= x0 else -abs(h)
    while (step > 0 and x < x_end - 1e-12) or (step < 0 and x > x_end + 1e-12):
        Y = rk4_step(x, Y, step)
        x += step
        xs.append(x)
        Ys.append(Y)
    return np.array(xs), np.array(Ys)

# Second-order Taylor polynomial about x0 = 3
# We derived: T2(x) = 6 + (x - 3) - (11/2)*(x - 3)**2
def T2(x):
    return 6.0 + (x - 3.0) - 0.5*11.0*(x - 3.0)**2

# Initial conditions at x = 3
x0 = 3.0
y0 = 6.0
yp0 = 1.0

# Choose interval around x = 3 for comparison
x_left = 2.0
x_right = 4.0
h = 0.001

# Solve ODE numerically from x0 down to x_left and up to x_right
xs_left, Ys_left = solve_ode_rk4(x0, x_left, -h, y0, yp0)
xs_right, Ys_right = solve_ode_rk4(x0, x_right, h, y0, yp0)

# Combine left (reversed to be increasing) and right
xs = np.concatenate((xs_left[::-1], xs_right[1:]))
Ys = np.concatenate((Ys_left[::-1], Ys_right[1:]))
y_true = Ys[:, 0]

# Evaluate Taylor polynomial and error
y_taylor = T2(xs)
error = np.abs(y_true - y_taylor)

print("Second-order Taylor polynomial: T2(x) = 6 + (x - 3) - (11/2)*(x - 3)^2")
print("Sample value at x = 3.5, T2(3.5) =", T2(3.5))

# Plot numerical solution and Taylor polynomial
plt.figure()
plt.plot(xs, y_true, label="Numerical solution (RK4)")
plt.plot(xs, y_taylor, '--', label="Second-order Taylor polynomial T2(x)")
plt.xlabel("x")
plt.ylabel("y(x)")
plt.title("Solution vs. Second-Order Taylor Polynomial about x = 3")
plt.grid(True)
plt.legend()

# Plot absolute error
plt.figure()
plt.plot(xs, error)
plt.xlabel("x")
plt.ylabel("|y(x) - T2(x)|")
plt.title("Error of Second-Order Taylor Approximation")
plt.grid(True)

plt.show()

