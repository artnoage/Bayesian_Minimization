import cupy as cp
from VisualisationBarycenter import *
from Algorithm import *
from Loglikelihood import *

NumberOfIterations=10
SampleSize=10000


#Provide the Images/Archetypes
A1=cp.concatenate((cp.ones(4), cp.zeros(12)), axis=0)
A1= A1 / cp.sum(A1)
A2=cp.concatenate((cp.zeros(12), cp.ones(4)), axis=0)
A2= A2 / cp.sum(A2)
Archetypes   = cp.array([A1, A2])


#Initializing Mean and Covariance

MeanMatrixInitialization, CovMatrixInitialization=initialisation(Archetypes, MeanMatrix="New",CovMatrix="New",Factor=10**(-4))



#Transformation is Sigmoid ,Exponential,Normalized Exponential and No Transformation. Penalty types are Square, Entropy and RevEntropy. PriorType= Gaussian.
# If you get an error regarding the weights, it means that the Loglikelihood values are big (remember that they go through exponential).
# If the algorithm stacks then restarting with a smaller Factor may help. Normalizing Covarance also helps some times.
# If you use any transformation change the factor to 1 and reduce until you start seeing decenting numbers.

MeanMatrix, CovMatrix = OneStep(Archetypes, TransformationFunction="Sigmoid", BarycenterPenalty="Square", ArchetypePenalty="Square",
                                ReluPenalty="Yes", PriorType="Gaussian", SampleSize=SampleSize, NumberOfIterations=NumberOfIterations,MeanMatrixInitialization=MeanMatrixInitialization,
                              CovMatrixInitialization=CovMatrixInitialization, NormalizeCovariance="Yes", LoglikelihoodFactor=0.001)


#Visualisation
#ani=visualise(Archetypes,Barycenter)
#plt.show()
