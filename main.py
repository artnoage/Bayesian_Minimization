import numpy as np
from VisualisationBarycenter import *
from Algorithm import *
from Loglikelihood import *

NumberOfIterations=100000
SampleSize=3000


#Provide the Images/Archetypes
A1=np.concatenate((np.ones(4), np.zeros(12)),axis=0)
A1=A1/np.sum(A1)
A2=np.concatenate((np.zeros(12),np.ones(4)),axis=0)
A2=A2/np.sum(A2)
Archetypes   = np.array([A1, A2])


#Initializing Mean and Covariance

MeanMatrixInitialization, CovMatrixInitialization=initialisation(Archetypes, MeanMatrix="Load",CovMatrix="New",Factor=10**(-5))



#Transformation is Normalized,Exponential,Normalized Exponential and No Transformation. Penalty types are Square, Entropy and RevEntropy. PriorType= Gaussian.
# If you get an error fro the weights, it means that the Loglikelihood values are big (remember that they go through exponential).

MeanMatrix, CovMatrix = OneStep(Archetypes, TransformationFunction="No", BarycenterPenalty="Square", ArchetypePenalty="Square",PriorType="Gaussian",
                              SampleSize=SampleSize, NumberOfIterations=NumberOfIterations,MeanMatrixInitialization=MeanMatrixInitialization,
                              CovMatrixInitialization=CovMatrixInitialization, LoglikelihoodFactor=0.5)


#Visualisation
#ani=visualise(Archetypes,Barycenter)
#plt.show()
