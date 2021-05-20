import numpy as np
from VisualisationBarycenter import *
from Algorithm import *
from Objective import *

NumberOfIterations=100000
SampleSize=1000


#Provide the Images/Archetypes
A1=np.concatenate((np.ones(4), np.zeros(12)),axis=0)
A1=A1/np.sum(A1)
A2=np.concatenate((np.zeros(12),np.ones(4)),axis=0)
A2=A2/np.sum(A2)
Archetypes   = np.array([A1, A2])


#Initializing Mean and Covariance

MeanMatrixInitialization, CovMatrixInitialization=initialisation(Archetypes, MeanMatrix="New",CovMatrix="New",Factor=10**(-4))



#Transformation is Normalized,Exponential,Normalized Exponential and No Transformation. Penalty types are Square, Entropy and RevEntropy. PriorType= Gaussian.

MeanMatrix, CovMatrix = OneStep(Archetypes, TransformationFunction="No", BarycenterPenalty="Square", ArchetypePenalty="Square",PriorType="Gaussian",
                              SampleSize=SampleSize, NumberOfIterations=NumberOfIterations,MeanMatrixInitialization=MeanMatrixInitialization,
                              CovMatrixInitialization=CovMatrixInitialization)


#Visualisation
#ani=visualise(Archetypes,Barycenter)
#plt.show()
