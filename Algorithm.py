from Objective import *
from Gaussians import *

def OneStep(Archetypes, TransformationFunction, BarycenterPenalty, ArchetypePenalty, PriorType, SampleSize, NumberOfIterations, MeanMatrixInitialization, CovMatrixInitialization):

    NumberOfAtoms  = len(Archetypes[0])
    MeanMatrix = MeanMatrixInitialization
    CovMatrix = CovMatrixInitialization
    for i in range(NumberOfIterations):
        Samples = SampleGeneration(PriorType, MeanMatrix, CovMatrix, SampleSize)
        LogLikelihoodValues = LogLikelihood(Samples, (Archetypes, TransformationFunction, BarycenterPenalty, ArchetypePenalty))
        Weights = np.exp(-LogLikelihoodValues)
        MeanMatrix = np.array(GaussianReconstruction(PriorType, Samples, Weights)[0])
        CovMatrix  = np.array(GaussianReconstruction(PriorType, Samples, Weights)[1])
        Factor=np.max(np.diag(CovMatrix))
        print(Factor)
        if i%500==499:
            CovMatrix = np.identity(171)/1000
        else:
            CovMatrix = CovMatrix
        Barycenter = np.array(Transformation([MeanMatrix[:NumberOfAtoms]], Inputtype="Barycenter",Transformationfunction=TransformationFunction))

        print("The minimum loglikelihood value is ", np.mean(LogLikelihoodValues[LogLikelihoodValues.argsort()[-10:][::-1]]), "\n")
        print("The mean barycenter is  ", Barycenter ,  "\n")

    with open('LogNormal.npy', 'wb') as f:
        np.save(f, MeanMatrix)
        np.save(f, CovMatrix)
    return MeanMatrix, CovMatrix
