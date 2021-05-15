from Objective import *

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
        if  i%30==0 and 0==0:
            Factor=np.max(np.diag(CovMatrix))
            Barycenter=np.array(Transformation([MeanMatrix[:NumberOfAtoms]],Inputtype="Barycenter", Transformationfunction=TransformationFunction))
            print("The minimum loglikelihood value is ", np.min(LogLikelihoodValues), "\n")
            print("The mean barycenter is  ", Barycenter ,  "\n")
            CovMatrix = np.max([Factor,10**(-6)]) * np.identity(171)
        if  i%900==899:
            with open('LogNormal.npy', 'wb') as f:
                np.save(f, MeanMatrix)
                np.save(f, CovMatrix)
    return MeanMatrix, CovMatrix
