import itertools
import numpy as np
from TransformationsAndPenalties import *

def LogLikelihood(Batch, args):

    Archetypes=args[0]
    PartitionLength= np.int(np.sqrt(len(Archetypes[0])))
    NumberOfAtoms = PartitionLength ** 2
    NumberOfArchetypes=len(Archetypes)

    #Creating the cost Matrix
    T = np.linspace(0, 1, PartitionLength)
    couples = np.array(np.meshgrid(T, T)).T.reshape(-1, 2)
    x = np.array(list(itertools.product(couples, repeat=2)))
    a = x[:, 0]
    b = x[:, 1]

    CostMatrix = np.linalg.norm(a - b, axis=1)**2
    CostMatrix = CostMatrix + np.zeros((len(Batch), NumberOfArchetypes, 1))
    Archetypes = Archetypes + np.zeros((len(Batch), NumberOfArchetypes, 1))



    #Isolating the Barycenter from the rest of the input
    Barycenter         = Batch[:,:NumberOfAtoms]

    # We apply the tranformation on the Barycenter.
    Barycenter         = Transformation(Barycenter,"Barycenter",args[1])

    # Creating a copy of the Barycenter candidate that will correspond to each Archetype

    Barycenter         = np.repeat([Barycenter],NumberOfArchetypes,axis=0).transpose(1, 0, 2)

    #Isolating the Plans from the rest of the input
    Plans              = Batch[:,NumberOfAtoms:]

    #Spliting The plans
    Plans              = np.array(np.split(Plans,NumberOfArchetypes,axis=1)).transpose(1,0,2)

    #We apply the tranformation on the plans
    Plans              = Transformation(Plans,"Plans",args[1])

    #Calculating the transportaion cost
    TransportationCost = np.sum(np.sum(CostMatrix[:,:,:]*Plans[:,:,:],axis=2),axis=1)

    #Calculating the Margins of each plan.
    BarycenterMargin   = np.sum(Plans.reshape(-1,NumberOfArchetypes,NumberOfAtoms,NumberOfAtoms).transpose(0,1,3,2),axis=3)
    ArchetypeMargin   = np.sum(Plans.reshape(-1,NumberOfArchetypes,NumberOfAtoms,NumberOfAtoms),axis=3)

    #Calculating the penalty for each plan.
    BarycenterPenalty = Penalty(BarycenterMargin,Barycenter,args[2])
    ArchetypePenalty  = Penalty(ArchetypeMargin,Archetypes,args[3])

    # This punishes negative values.

    if args[1]!= "Normalized Exponential":
        ReluData = -np.sum(np.minimum(Batch, np.zeros((len(Batch),len(Batch[0])))), axis=1)

    else :
        ReluData=0


    TotalCost = TransportationCost + 10*ArchetypePenalty + 10*BarycenterPenalty +100*ReluData

    return TotalCost







