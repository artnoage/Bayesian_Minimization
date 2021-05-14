import numpy as np
from VisualisationBarycenter import *
from Algorithm import *
from Objective import *

NumberOfIterations=1000
SampleSize=10000

#Provide the Images
Archetypes   = np.array([[0.3, 0.3,0.3,0.01,0.01,0.02,0.02,0.02,0.02], [0.02,0.02,0.02,0.02,0.01,0.01,0.3,0.3,0.3]])
NumberOfAtoms = len(Archetypes[0])
NumberOfArchetypes = len(Archetypes)
PlanSize = NumberOfAtoms ** 2
TotalDimension = NumberOfAtoms + NumberOfArchetypes * PlanSize
StepSize="Natural"


#Initializing Mean and Covariance

#MeanMatrixInitialization = np.zeros(TotalDimension)
#CovMatrixInitialization = np.identity(TotalDimension)

#with open('LogNormal.npy', 'rb') as g:
#    MeanMatrixInitialization=np.load(g)
#    CovMatrixInitialization=np.load(g)

MeanMatrixInitialization=np.zeros(TotalDimension)
CovMatrixInitialization=np.identity(TotalDimension)

#Itterating Algorithm

#Transformation is Normalized,Exponential,Normalized Exponential and No Transformation. Penalty types are Square, Entropy and RevEntropy. PriorType= Gaussian, LogNormal.


MeanMatrix, CovMatrix = OneStep(Archetypes, TransformationFunction="No", BarycenterPenalty="Square", ArchetypePenalty="Square",PriorType="Gaussian",
                              SampleSize=SampleSize, NumberOfIterations=NumberOfIterations,MeanMatrixInitialization=MeanMatrixInitialization,
                              CovMatrixInitialization=CovMatrixInitialization,StepSize=StepSize)

#MeanMatrix, CovMatrix =OneStep(Archetypes, TransformationFunction="No", BarycenterPenalty="RevEntropy", ArchetypePenalty="RevEntropy",PriorType="Gaussian",
# SampleSize=SampleSize, NumberOfIterations=NumberOfIterations,MeanMatrixInitialization=MeanMatrixInitialization,CovMatrixInitialization=CovMatrixInitialization,StepSize=StepSize)



with open('LogNormal.npy', 'wb') as f:
    np.save(f,MeanMatrix)
    np.save(f,CovMatrix)

#Visualisation
#ani=visualise(Archetypes,Barycenter)
#plt.show()
