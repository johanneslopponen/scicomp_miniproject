import numpy as np
import matplotlib.pyplot as plt

def euler(N,T,h,D,u,num_steps,checkpoints):
    X = np.zeros((N,2))

    t = np.linspace(0,T, num_steps + 1)

    snapshots = {}

    for n in range(0, num_steps+1):
        Z = np.random.randn(N,2)

        X = X + u * h + np.sqrt(2 * D * h) * Z
        
        if round(t[n],1) in checkpoints:
            snapshots[int(t[n])] = X.copy()

    return snapshots



N = 2000         
T = 60       
h = 0.1         
D = 0.02         
u = np.array([0.3, 0]) 
num_steps = int(T / h)
checkpoints = [15, 30, 45, 60]

snapshots = euler(N,T,h,D,u,num_steps, checkpoints)


fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
axes = axes.flatten()

for i, t in enumerate(checkpoints):
    pos = snapshots[t]
    axes[i].scatter(pos[:, 0], pos[:, 1], s=1, alpha=0.5)
    axes[i].set_title(f"Tid: {t}s")
    axes[i].set_xlim(0, 25) # Domängräns x 
    axes[i].set_ylim(-5, 5) # Domängräns y 
    axes[i].set_xlabel("x")
    axes[i].set_ylabel("y")

plt.tight_layout()
plt.show()