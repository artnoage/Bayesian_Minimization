import numpy as np
from VisualisationBarycenter import *
from Algorithm import *
from Objective import *

NumberOfIterations=10
SampleSize=100000

#Provide the Images
Archetypes   = np.array([[0.3, 0.3,0.3,0.01,0.01,0.02,0.02,0.02,0.02], [0.02,0.02,0.02,0.02,0.01,0.01,0.3,0.3,0.3]])
NumberOfAtoms = len(Archetypes[0])
NumberOfArchetypes = len(Archetypes)
PlanSize = NumberOfAtoms ** 2
TotalDimension = NumberOfAtoms + NumberOfArchetypes * PlanSize



#Initializing Mean and Covariance

#MeanMatrixInitialization = np.zeros(TotalDimension)
#

if 1==0:
    with open('LogNormal.npy', 'rb') as g:
        MeanMatrixInitialization=np.load(g)
        #CovMatrixInitialization=np.load(g)
        CovMatrixInitialization = np.identity(TotalDimension)
else:
    MeanMatrixInitializationPart1=np.ones(NumberOfAtoms)/NumberOfAtoms
    MeanMatrixInitializationPart2=np.ones((NumberOfAtoms**2)*NumberOfArchetypes)/(NumberOfAtoms**2)
    MeanMatrixInitialization=np.concatenate((MeanMatrixInitializationPart1,MeanMatrixInitializationPart2),axis=0)
    CovMatrixInitialization = np.identity(TotalDimension)

#Itterating Algorithm

#Transformation is Normalized,Exponential,Normalized Exponential and No Transformation. Penalty types are Square, Entropy and RevEntropy. PriorType= Gaussian, LogNormal.


MeanMatrix, CovMatrix = OneStep(Archetypes, TransformationFunction="No", BarycenterPenalty="Square", ArchetypePenalty="Square",PriorType="Hybrid",
                              SampleSize=SampleSize, NumberOfIterations=NumberOfIterations,MeanMatrixInitialization=MeanMatrixInitialization,
                              CovMatrixInitialization=CovMatrixInitialization)


#Visualisation
#ani=visualise(Archetypes,Barycenter)
#plt.show()
