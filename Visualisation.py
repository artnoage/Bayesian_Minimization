import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from scipy.stats import multivariate_normal

def visualise(Likelihood,Mean,Covariance):
    N = 200 # Meshsize
    fps =3 # frame per sec
    frn = len(Mean)*fps # frame number of the animation
    T = np.linspace(-20,20,N+1)
    x, y = np.meshgrid(T, T)

    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x
    pos[:, :, 1] = y


    # Coloring
    fig = plt.figure()
    ax= fig.add_subplot(121,projection='3d')
    ax2 = fig.add_subplot(122)
    colors=[]
    colors.append(plt.cm.viridis(0.25))
    colors.append(plt.cm.inferno(0.25))
    colors.append(plt.cm.inferno(0.5))
    shades=[]
    shades.append(plt.cm.Reds)
    shades.append(plt.cm.Blues)

    # Update
    Distributions = []
    for i in range(len(Mean)):
        for j in range(fps):
            Distributions.append(multivariate_normal(Mean[i], Covariance[i]))
    def update_plot(frame_number, Distributions):
        ax.clear()
        ax2.clear()
        ax.plot_wireframe(x, y, Likelihood([x,y]), color=colors[0], alpha=0.5, rstride=10, cstride=10)
        ax.plot_wireframe(x, y, Distributions[frame_number].pdf(pos),color=colors[2], rstride=10, cstride=10, alpha=0.3)
        ax.view_init(10 * np.cos(0.03 * frame_number) + 15, 10 * np.sin(0.03 * frame_number) + 15)
        ax2.pcolormesh(x, y, Likelihood([x, y]), cmap=shades[1], shading='auto', alpha=0.3)
        ax2.pcolormesh(x, y,Distributions[frame_number].pdf(pos), cmap=shades[0], shading='auto',alpha=0.3)

    return animation.FuncAnimation(fig, update_plot, frn, fargs=(Distributions,), interval=1000 / fps)