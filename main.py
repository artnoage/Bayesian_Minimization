import numpy as np
from Visualisation import *
from Algorithm import *
from Objective import *

NumberOfIterations=100


#Provide the Images
Archetypes   = np.array([[0.92, 0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01], [0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.92]])


#Itterating Algorithm
#Transformation is Normalized,Exponential,Normalized Exponential and No Transformation. Penalty types are Square and Entropy.
OneStep(Archetypes, Transformation="Exponential Normalized", BarycenterPenalty="Square", ArchetypePenalty="Square", SampleSize=10**5, NumberOfIterations=NumberOfIterations)
OneStep(Archetypes, Transformation="Exponential Normalized", BarycenterPenalty="Entropy", ArchetypePenalty="Entropy", SampleSize=10**5, NumberOfIterations=NumberOfIterations)
OneStep(Archetypes, Transformation="Normalized", BarycenterPenalty="Square", ArchetypePenalty="Square", SampleSize=10**5, NumberOfIterations=NumberOfIterations)
OneStep(Archetypes, Transformation="Exponential", BarycenterPenalty="Entropy", ArchetypePenalty="Entropy", SampleSize=10**5, NumberOfIterations=NumberOfIterations)
OneStep(Archetypes, Transformation="No", BarycenterPenalty="Square", ArchetypePenalty="Square", SampleSize=10**5, NumberOfIterations=NumberOfIterations)
OneStep(Archetypes, Transformation="Normalized", BarycenterPenalty="Entropy", ArchetypePenalty="Entropy", SampleSize=10**5, NumberOfIterations=NumberOfIterations)


#Visualisation
#ani=visualise(Objective,MeanMatrices,CovMatrices)
#plt.show()