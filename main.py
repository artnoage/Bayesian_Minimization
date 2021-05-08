import numpy as np
from Visualisation import *
from Algorithm import *
from Objective import *

#Define parameters
PartitionLength    = 3
Archetypes1        = np.array([[1, 0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,1]])
Archetypes         = Archetypes1
NumberOfArchetypes = len(Archetypes)
MicroSampleSize    = 10000
NumberOfAtoms      = PartitionLength**2
PlanSize           = NumberOfAtoms**2
TotalDimension     = NumberOfAtoms + NumberOfArchetypes * PlanSize


print(3)
#Initial Matrix
#MeanMatrix          = np.array(np.random.rand(TotalDimension))
CovMatrix            = np.identity(TotalDimension)



def Likelihood(Sample):
    return Objective(Sample, PartitionLength, Archetypes,Transformation)


#Itterating Algorithm

MeanMatrix = np.array(np.random.rand(TotalDimension))
OneStep(MeanMatrix, CovMatrix, MicroSampleSize, Likelihood,10)


#Visualisation
#ani=visualise(Objective,MeanMatrices,CovMatrices)
#plt.show()