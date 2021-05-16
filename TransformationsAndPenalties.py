import numpy as np

def Transformation(Batch,Inputtype,Transformationfunction):
    if Inputtype == "Barycenter" and Transformationfunction == "Normalized Exponential":
        Batch = np.exp(Batch)
        BatchNorm = np.sum(Batch, axis=1, keepdims=True)
        Batch = Batch / BatchNorm
    elif Inputtype == "Plans" and Transformationfunction == "Normalized Exponential":
        Batch = np.exp(Batch)
        BatchNorm = np.sum(Batch, axis=2, keepdims=True)
        Batch = Batch / BatchNorm
    elif Inputtype=="Barycenter" and Transformationfunction=="Exponential":
        Batch=np.exp(Batch)
    elif Inputtype=="Plans" and Transformationfunction=="Exponential":
        Batch=np.exp(Batch)
    elif Inputtype=="Barycenter" and Transformationfunction=="Normalized":
        BatchNorm=np.sum(Batch,axis=1,keepdims=True)
        Batch=Batch/BatchNorm
    elif Inputtype=="Plans" and Transformationfunction=="Normalized":
        BatchNorm=np.sum(Batch,axis=2,keepdims=True)
        Batch=Batch/BatchNorm
    return Batch

def Penalty(FirstConfiguration,SecondConfiguration,Metric):
    if Metric=="Square":
        Penalty = np.sum(np.linalg.norm(FirstConfiguration -SecondConfiguration,axis=2),axis=1)
    if Metric=="Entropy":
        Penalty = np.sum(np.sum(np.abs(FirstConfiguration)*(np.log(np.abs(FirstConfiguration))-np.log(np.abs(SecondConfiguration))),axis=2),axis=1)
    if Metric == "RevEntropy":
        Penalty = np.sum(np.sum(np.abs(SecondConfiguration) * (np.log(np.abs(SecondConfiguration)) - np.log(np.abs(FirstConfiguration))),axis=2), axis=1)

    return Penalty