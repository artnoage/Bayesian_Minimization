import numpy as np
import time as time
from VisualisationBarycenter import *
from Algorithm import *
from Loglikelihood import *

NumberOfIterations=1000
SampleSize=100


#Provide the Images/Archetypes
A1=np.concatenate((np.ones(4), np.zeros(12)),axis=0)
A1=A1/np.sum(A1)
A2=np.concatenate((np.zeros(12),np.ones(4)),axis=0)
A2=A2/np.sum(A2)
#Archetypes   = np.array([A1, A2])
Archetypes  = np.array([A1, A2])

#Initializing Mean and Covariance

MeanMatrixInitialization, CovMatrixInitialization=initialisation(Archetypes, MeanMatrix="New",CovMatrix="New",Factor=10**(-2))



#Transformation is Sigmoid ,Exponential,Normalized Exponential and No Transformation. Penalty types are Square, Entropy and RevEntropy. PriorType= Gaussian.
# If you get an error regarding the weights, it means that the Loglikelihood values are big (remember that they go through exponential).
# If the algorithm stacks then restarting with a smaller Factor may help. Normalizing Covarance also helps some times.
# If you use any transformation change the factor to 1 and reduce until you start seeing decenting numbers.

MeanMatrix, CovMatrix = OneStep(Archetypes, TransformationFunction="Sigmoid", BarycenterPenalty="Square", ArchetypePenalty="Square",
                                ReluPenalty="Yes", PriorType="Gaussian", SampleSize=SampleSize, NumberOfIterations=NumberOfIterations,MeanMatrixInitialization=MeanMatrixInitialization,
                              CovMatrixInitialization=CovMatrixInitialization, NormalizeCovariance="Yes", LoglikelihoodFactor=1)


#Visualisation
#ani=visualise(Archetypes,Barycenter)
#plt.show()
