import numpy as np
from Visualisation import *
from Algorithm import *
from Objective import *

#Define parameters
PartitionLength    = 2
Archetypes         = np.array([[1, 0, 0, 0], [0,0 , 1, 0]])
NumberOfArchetypes = len(Archetypes)
MicroSampleSize    = 1000000
NumberOfAtoms      = PartitionLength**2
PlanSize           = NumberOfAtoms**2
TotalDimension     = NumberOfAtoms + NumberOfArchetypes * PlanSize



#Initial Matrix
#[-1,-100,-1,-100,-1,-100,-1,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-1,-100,-1,-100,-100,-100,-100,-100]
#[0.5,0,0.5,0,0.5,0,0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.5,0,0.5,0,0,0,0,0]
MeanMatrix          = np.array(np.random.rand(TotalDimension))
CovMatrix           = np.identity(TotalDimension)



def Likelihood(Sample):
    return ObjectiveRelu(Sample, PartitionLength, Archetypes)


#Itterating Algorithm

for i in range (1):
    MeanMatrix = np.array(np.random.rand(TotalDimension))
    OneStep(MeanMatrix, CovMatrix, MicroSampleSize, Likelihood, 50)


#Visualisation
#ani=visualise(Objective,MeanMatrices,CovMatrices)
#plt.show()