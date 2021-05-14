from Objective import *

def OneStep(Archetypes, TransformationFunction, BarycenterPenalty, ArchetypePenalty, PriorType, SampleSize, NumberOfIterations, MeanMatrixInitialization, CovMatrixInitialization):

    NumberOfAtoms  = len(Archetypes[0])
    MeanMatrix = MeanMatrixInitialization
    CovMatrix = CovMatrixInitialization
    Last=np.zeros(10)+500
    LogLikelihoodFactor=1
    for i in range(NumberOfIterations):
        Samples = SampleGeneration(PriorType, MeanMatrix, CovMatrix, SampleSize)
        LogLikelihoodValues = LogLikelihood(Samples, (Archetypes, TransformationFunction, BarycenterPenalty, ArchetypePenalty))
        Last=np.append(Last,[np.min(LogLikelihoodValues)])
        Last=np.delete(Last,[0])
        LastAverage=np.mean(Last)
        if LogLikelihoodFactor*LastAverage<200:
            Last = np.zeros(10) + 500
            LogLikelihoodFactor=2*LogLikelihoodFactor
        Weights = np.exp(-LogLikelihoodFactor*LogLikelihoodValues)

        MeanMatrix = np.array(GaussianReconstruction(PriorType, Samples, Weights)[0])
        CovMatrix  = np.array(GaussianReconstruction(PriorType, Samples, Weights)[1])
        if  (i%20==0):
            Barycenter=np.array(Transformation([MeanMatrix[:NumberOfAtoms]],Inputtype="Barycenter", Transformationfunction=TransformationFunction))
            print("The minimum loglikelihood value is ", np.min(LogLikelihoodValues), "\n")
            print("The mean barycenter is  ", Barycenter ,  "\n")
            CovMatrix = 10 ** (-5) * np.identity(171)
    return MeanMatrix, CovMatrix
