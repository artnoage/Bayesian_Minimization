import numpy as np
from Visualisation import *
from Algorithm import *
from Objective import *



#Provide the Images
Archetypes   = np.array([[1, 0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,1]])


#Itterating Algorithm
OneStep(Archetypes,Transformation="Les",BarycenterPenalty="Entropy",ArchetypePenalty="Square",SampleSize=2*10**5,NumberOfIterations=100)


#Visualisation
#ani=visualise(Objective,MeanMatrices,CovMatrices)
#plt.show()