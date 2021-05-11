import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from scipy.stats import multivariate_normal

def visualise(Archetypes,Barycenter):
    PartitionLength=np.int(np.sqrt(len(Archetypes[0])))

    Archetypes= -Archetypes.reshape(len(Archetypes),PartitionLength,PartitionLength)
    Barycenter= -Barycenter.reshape(PartitionLength,PartitionLength)
    Archetypes = Archetypes[::-1]
    Barycenter = Barycenter[::-1]
    ax=[]
    FactorA=np.int(np.sqrt(len(Archetypes)))
    FactorB=np.int(len(Archetypes)/FactorA)
    for i in range(len(Archetypes)+2):
        ax.append([])
    fig=plt.figure(figsize=(12,12))
    print(FactorA,FactorB)
    for i in range(len(Archetypes)):
        ax[i]=fig.add_subplot(FactorA+1,FactorB,i+1)
        ax[i].imshow(Archetypes[i],cmap="viridis")
    ax[len(Archetypes)+1]=fig.add_subplot(FactorA+1,1,FactorA+1)
    ax[len(Archetypes)+1].imshow(Barycenter,cmap="viridis")

    plt.show()