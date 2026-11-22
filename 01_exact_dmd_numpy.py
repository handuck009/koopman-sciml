import numpy as np
import matplotlib.pyplot as plt

nx = 200
nt = 120

x = np.linspace(-10, 10, nx)
t = np.linspace(0, 8, nt)
dt = t[1] - t[0]

phi1 = np.exp(-(x + 3)**2 / 4)
phi2 = np.sin(2 * x) * np.exp(-x**2 / 20)

omega1 = 2.0
omega2 = 5.0
sigma1 = -0.05
sigma2 = -0.10

X_data = np.zeros((nx, nt))

for k, tk in enumerate(t):
    X_data[:, k] = (
        phi1 * np.cos(omega1 * tk) * np.exp(sigma1 * tk)
        + 0.5 * phi2 * np.sin(omega2 * tk) * np.exp(sigma2 * tk)
    )

plt.figure(figsize=(8, 4))
plt.imshow(X_data, aspect="auto", origin="lower", extent=[t[0], t[-1], x[0], x[-1]])
plt.colorbar(label="u(x,t)")
plt.xlabel("t")
plt.ylabel("x")
plt.title("Synthetic flow-like data")
plt.show()
