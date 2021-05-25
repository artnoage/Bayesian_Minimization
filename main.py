import cupy as cp
from VisualisationBarycenter import *
from Algorithm import *
from Loglikelihood import *

NumberOfIterations=100000
SampleSize=1000


#Provide the Images/Archetypes

A1=cp.concatenate((cp.ones(3), cp.zeros(6)), axis=0)
A1= A1 / cp.sum(A1)
A2=cp.concatenate((cp.zeros(6), cp.ones(3)), axis=0)
A2= A2 / cp.sum(A2)
Archetypes  = cp.array([A1, A2,A1, A2,A1, A2, A1,A2,A2,A1,A1,A1])
#B=cp.random.randint(0,2,10)
#Archetypes=cp.take(Archetypes,B,0)
#print(Archetypes[0])
#Initializing Mean and Covariance

MeanMatrixInitialization, CovMatrixInitialization=initialisation(Archetypes, MeanMatrix="New",CovMatrix="New",Factor=1)



#Transformation is Sigmoid ,Exponential,Normalized Exponential and No Transformation. Penalty types are Square, Entropy and RevEntropy. PriorType= Gaussian.
# If you get an error regarding the weights, it means that the Loglikelihood values are big (remember that they go through exponential).
# If the algorithm stacks then restarting with a smaller Factor may help. Normalizing Covarance also helps some times.
# If you use any transformation change the factor to 1 and reduce until you start seeing decenting numbers.

MeanMatrix, CovMatrix = OneStep(Archetypes, TransformationFunction="Sigmoid", BarycenterPenalty="Square", ArchetypePenalty="Square",
                                ReluPenalty="Yes", PriorType="Gaussian", SampleSize=SampleSize, NumberOfIterations=NumberOfIterations,MeanMatrixInitialization=MeanMatrixInitialization,
                              CovMatrixInitialization=CovMatrixInitialization, NormalizeCovariance="No", LoglikelihoodFactor=0.1)


#Visualisation
#ani=visualise(Archetypes,Barycenter)
#plt.show()
