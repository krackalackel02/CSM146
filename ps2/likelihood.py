import numpy as np
import matplotlib.pyplot as plt


X = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0])  
n = len(X)
S = np.sum(X)  


def likelihood(theta, S, n):
    return (theta ** S) * ((1 - theta) ** (n - S))


theta_values = np.linspace(0, 1, 101)
likelihood_values = [likelihood(theta, S, n) for theta in theta_values]


theta_mle = S / n  


plt.figure(figsize=(8, 5))
plt.plot(theta_values, likelihood_values, label=r'$L(\theta)$', color='blue')
plt.axvline(x=theta_mle, linestyle='--', color='red', label=r'$\hat{\theta}_{MLE}$')
plt.xlabel(r'$\theta$')
plt.ylabel(r'$L(\theta)$')
plt.title('Likelihood Function')
plt.legend()
plt.grid()
plt.show()



datasets = [
    {"n": 5, "S": 3, "label": "n = 5, S = 3"},
    {"n": 100, "S": 60, "label": "n = 100, S = 60"},
    {"n": 10, "S": 5, "label": "n = 10, S = 5"},
]


theta_values = np.linspace(0, 1, 101)


plt.figure(figsize=(12, 4))

for i, data in enumerate(datasets, 1):
    likelihood_values = [likelihood(theta, data["S"], data["n"]) for theta in theta_values]
    theta_mle = data["S"] / data["n"]  

    plt.subplot(1, 3, i)
    plt.plot(theta_values, likelihood_values, label=r'$L(\theta)$', color='blue')
    plt.axvline(x=theta_mle, linestyle='--', color='red', label=r'$\hat{\theta}_{MLE}$')
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$L(\theta)$')
    plt.title(data["label"])
    plt.legend()
    plt.grid()

plt.tight_layout()
plt.show()