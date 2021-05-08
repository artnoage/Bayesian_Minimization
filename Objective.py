import numpy as np
import itertools
from Gaussians import *

def Transformation(Batch,Inputtype,Transformationfunction):
    if Inputtype=="Barycenter":
        Batch=np.exp(Batch)
        BatchNorm=np.sum(Batch,axis=1,keepdims=True)
        Batch=Batch/BatchNorm
    if Inputtype=="Plans":
        Batch=np.exp(Batch)
        BatchNorm=np.sum(Batch,axis=2,keepdims=True)
        Batch=Batch/BatchNorm
    return Batch

def Penalty(FirstConfiguration,SecondConfiguration,Metric):
    if Metric=="Square":
        return np.sum(np.linalg.norm(FirstConfiguration -SecondConfiguration,axis=2),axis=1)
    if Metric=="Entropy":
        A=np.sum(np.sum(FirstConfiguration*(np.log(FirstConfiguration)-np.log(SecondConfiguration)),axis=2),axis=1)
        return A

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
    Barycenter         = np.array([Barycenter, Barycenter]).transpose(1, 0, 2)

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
    ReluData = np.sum(np.minimum(Batch - 0.001, 0), axis=1)

    TotalCost = TransportationCost + 5*ArchetypePenalty + 5*BarycenterPenalty

    return TotalCost







