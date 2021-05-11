import numpy as np
from Gaussians import *
from Objective import *

def OneStep(Archetypes, TransformationFunction, BarycenterPenalty, ArchetypePenalty, SampleSize, NumberOfIterations):
    NumberOfAtoms  = len(Archetypes[0])
    NumberOfArchetypes =len(Archetypes)
    PlanSize = NumberOfAtoms ** 2
    TotalDimension = NumberOfAtoms + NumberOfArchetypes * PlanSize


    MeanMatrix = np.array(np.random.rand(TotalDimension))
    CovMatrix = np.identity(TotalDimension)

    for i in range(NumberOfIterations):
        Samples = SampleGeneration(MeanMatrix, CovMatrix, SampleSize)
        LogLikelihoodValues = 0.1*LogLikelihood(Samples, (Archetypes, TransformationFunction, BarycenterPenalty, ArchetypePenalty))
        Weights = np.exp(-LogLikelihoodValues)

        MeanMatrix = np.array(GaussianReconstruction(Samples, Weights)[0])
        CovMatrix  = np.array(GaussianReconstruction(Samples, Weights)[1])
        #CovMatrix = np.identity(TotalDimension)
        factor=np.min(np.diag(CovMatrix))
        #print(factor)
        CovMatrix=0.0005*CovMatrix/factor
        Barycenter=np.array(Transformation([MeanMatrix[:NumberOfAtoms]],Inputtype="Barycenter", Transformationfunction=TransformationFunction))
        print("The minimum loglikelihood value is ",np.min(LogLikelihoodValues),"\n")
    print("The mean barycenter is  ", Barycenter ,  "\n")
    #print(BarycenterPenalty,"\n","\n")
    return Barycenter
