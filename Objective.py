import numpy as np
import itertools
from Gaussians import *

def Transformation(Batch):
    Batch=np.exp(Batch)
    return Batch

def Objective(Batch, Archetypes,Transformation):
    PartitionLength= np.int(np.sqrt(np.sqrt(len(Archetypes[0]))))
    NumberOfAtoms = PartitionLength ** 2
    PlanSize = NumberOfAtoms ** 2
    NumberOfArchetypes=len(Archetypes)
    T = np.linspace(0, 1, PartitionLength)
    couples = np.array(np.meshgrid(T, T)).T.reshape(-1, 2)
    x = np.array(list(itertools.product(couples, repeat=2)))
    a = x[:, 0]
    b = x[:, 1]

    CostMatrix = np.linalg.norm(a - b, axis=1)**2
    CostMatrix = CostMatrix + np.zeros((len(Batch), NumberOfArchetypes, 1))
    Archetypes = Archetypes + np.zeros((len(Batch), NumberOfArchetypes, 1))

    Batch               = np.exp(Batch)
    Barycenter         = np.array([Batch[:, :NumberOfAtoms], Batch[:, :NumberOfAtoms]]).transpose(1, 0, 2)
    BarycenterNorm     = np.array([[np.sum(Sample[0],keepdims=True),np.sum(Sample[1],keepdims=True)]for Sample in Barycenter])
    Barycenter         = Barycenter[:,:]/BarycenterNorm[:,:]
    Plans              = Batch[:, NumberOfAtoms:].reshape(-1, NumberOfArchetypes, PlanSize)
    PlansNorm          = np.array([[ np.sum(Sample[0],keepdims=True),np.sum(Sample[1],keepdims=True) ]for Sample in Plans])
    Plans = Plans / PlansNorm

    TransportationCost = np.sum(np.sum(CostMatrix[:,:,:]*Plans[:,:,:],axis=2),axis=1)
    BarycenterMargin  = np.sum(Plans.reshape(-1,NumberOfArchetypes,NumberOfAtoms,NumberOfAtoms).transpose(0,1,3,2),axis=3)

    ArchetypeMargin   = np.sum(Plans.reshape(-1,NumberOfArchetypes,NumberOfAtoms,NumberOfAtoms),axis=3)
    BarycenterPenalty = np.sum(np.linalg.norm(BarycenterMargin -Barycenter,axis=2),axis=1)
    ArchetypePenalty  = np.sum(np.linalg.norm(ArchetypeMargin -Archetypes,axis=2),axis=1)

    ReluData = np.sum(np.minimum(Batch - 0.001, 0), axis=1)
    TotalCost = TransportationCost + 5 * ArchetypePenalty + 5 * BarycenterPenalty
    return TotalCost







