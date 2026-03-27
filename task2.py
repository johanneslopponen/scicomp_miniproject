import numpy as np
import matplotlib.pyplot as plt

# Task 2

epsilon = 0.1
N = 2000         
T = 60       
h = 0.1         
D = 0.02 
u = np.array([0.3, 0]) 
num_steps = int(T / h)
checkpoints = [15, 30, 45, 60]


def euler_at_time(N, T, h, D, u):
    # Beräkna hur många steg som krävs för att nå tidpunkten T
    num_steps = int(T / h)
    
    # Initiera positionerna (N partiklar, 2 dimensioner)
    X = np.zeros((N, 2))

    # Förberäkna diffusionskonstanten för att spara beräkningskraft
    diffusion_scale = np.sqrt(2 * D * h)
    drift = u * h

    for _ in range(num_steps):
        Z = np.random.randn(N, 2)
        # Euler-Maruyama steg: X = X + u*dt + sqrt(2*D*dt)*Z
        X = X + drift + diffusion_scale * Z

    return X

X = euler_at_time(N,T,h,D,u)

def delta_eps(dist, epsilon):
    fac = 1/(2*np.pi*epsilon**2)
    e = np.exp(-dist / (2*epsilon**2))
    return fac * e

nx, ny = (200, 100)
x = np.linspace(0, 25, nx)
y = np.linspace(-5, 5, ny)
x_grid, y_grid = np.meshgrid(x, y)

def compute_concentration(x_grid, y_grid, snapshot, epsilon):
    N = snapshot.shape[0] #Antalet partiklar vid tiden t
    C = np.zeros(x_grid.shape)

    for k in range(N):
        dist = ((x_grid - snapshot[k,0])**2 + (y_grid - snapshot[k,1])**2)
        C += delta_eps(dist, epsilon)

    return C / N



t = np.array([15,30,45,60])
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
axes = axes.flatten()
vmin, vmax = 0, 0.2

for i in range(len(t)):
    
    
    X = euler_at_time(N, t[i], h, D, u)
    C = compute_concentration(x_grid, y_grid, X, epsilon)
    contour = axes[i].contourf(x_grid, y_grid, C, levels=50, 
                               cmap='Reds', vmin=vmin, vmax=vmax)
    axes[i].set_title(f"Tid: {t[i]} s")
    axes[i].set_aspect('equal')

fig.subplots_adjust(bottom=0.15)
cbar_ax = fig.add_axes([0.15, 0.08, 0.7, 0.03])
fig.colorbar(contour, cax=cbar_ax, orientation='horizontal', label='Koncentration $C(x,y,t)$')

plt.show()