import numpy as np
from Gaussians import *
from Objective import *

def OneStep(Archetypes, TransformationFunction, BarycenterPenalty, ArchetypePenalty, PriorType, SampleSize, NumberOfIterations, MeanMatrixInitialization, CovMatrixInitialization,StepSize):

    NumberOfAtoms  = len(Archetypes[0])
    MeanMatrix = MeanMatrixInitialization
    CovMatrix = CovMatrixInitialization

    for i in range(NumberOfIterations):
        Samples = SampleGeneration(PriorType, MeanMatrix, CovMatrix, SampleSize)
        LogLikelihoodValues = LogLikelihood(Samples, (Archetypes, TransformationFunction, BarycenterPenalty, ArchetypePenalty))
        print("The minimum loglikelihood value is ", np.min(LogLikelihoodValues), "\n")
        LogLikelihoodValues = 600*LogLikelihoodValues/np.min(LogLikelihoodValues)
        Weights = np.exp(-LogLikelihoodValues)

        MeanMatrix = np.array(GaussianReconstruction(PriorType, Samples, Weights)[0])
        CovMatrix  = np.array(GaussianReconstruction(PriorType, Samples, Weights)[1])
        if StepSize!="Natural":
            factor=np.min(np.diag(CovMatrix))
            CovMatrix=StepSize*CovMatrix/factor
        Barycenter=np.array(Transformation([MeanMatrix[:NumberOfAtoms]],Inputtype="Barycenter", Transformationfunction=TransformationFunction))

    print("The mean barycenter is  ", Barycenter ,  "\n")
    return MeanMatrix, CovMatrix
