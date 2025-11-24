import numpy as np
import matplotlib.pyplot as plt

# ODE: y'' - 2*x*y' + x**2*y = 0
# Rewrite as a first-order system:
#   y1' = y2
#   y2' = 2*x*y2 - x**2*y1

def f(x, Y):
    y1, y2 = Y
    dy1 = y2
    dy2 = 2*x*y2 - x**2*y1
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
    while x < x_end - 1e-10:
        Y = rk4_step(x, Y, h)
        x += h
        xs.append(x)
        Ys.append(Y)
    return np.array(xs), np.array(Ys)

# Taylor-series coefficients via recurrence
def taylor_coeffs(a0, a1, N):
    a = np.zeros(N+3, dtype=float)
    a[0] = a0
    a[1] = a1
    a[2] = 0.0
    a[3] = a1 / 3.0

    for n in range(2, N+1):
        a[n+2] = (2*n*a[n] - a[n-2]) / ((n+2)*(n+1))
    return a[:N+1]

def taylor_eval(x, coeffs):
    powers = np.array([x**n for n in range(len(coeffs))])
    return np.dot(coeffs, powers)

a0 = 1.0
a1 = -1.0

N_max = 10
coeffs = taylor_coeffs(a0, a1, N_max)

N_trunc = 4
y_approx_35 = taylor_eval(3.5, coeffs[:N_trunc+1])
print("4th-order Taylor approximation y(3.5) =", y_approx_35)

x0 = 0.0
y0 = 1.0
yp0 = -1.0
x_end = 4.0
h = 0.001
xs, Ys = solve_ode_rk4(x0, x_end, h, y0, yp0)
y_true = Ys[:,0]

orders = [2, 4, 6, 8, 10]

plt.figure()
plt.plot(xs, y_true, label="RK4 numerical solution")

for N in orders:
    cN = coeffs[:N+1]
    y_taylor = np.array([taylor_eval(x, cN) for x in xs])
    plt.plot(xs, y_taylor, linestyle="--", label=f"Taylor, N={N}")

plt.xlabel("x")
plt.ylabel("y(x)")
plt.title("Taylor Series Approximation and Convergence")
plt.legend()
plt.grid(True)
plt.show()

