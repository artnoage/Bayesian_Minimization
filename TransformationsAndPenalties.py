import numpy as np

def Transformation(Sample, Inputtype, Transformationfunction):
    Batchdims = np.ndim(Sample)-1
    if Inputtype == "Barycenter" and Transformationfunction == "Sigmoid":
        Sample = np.exp(Sample)
        BatchNorm = np.sum(Sample, axis=Batchdims, keepdims=True)
        Sample = Sample / BatchNorm
    elif Inputtype == "Plans" and Transformationfunction == "Sigmoid":
        Sample = np.exp(Sample)
        BatchNorm = np.sum(Sample, axis=Batchdims, keepdims=True)
        Sample = Sample / BatchNorm
    elif Inputtype=="Barycenter" and Transformationfunction=="Exponential":
        Sample=np.exp(Sample)
    elif Inputtype=="Plans" and Transformationfunction=="Exponential":
        Sample=np.exp(Sample)
    elif Inputtype=="Barycenter" and Transformationfunction=="Normalized":
        BatchNorm=np.sum(Sample, axis=Batchdims, keepdims=True)
        Sample= Sample / BatchNorm
    elif Inputtype=="Plans" and Transformationfunction=="Normalized":
        BatchNorm=np.sum(Sample, axis=Batchdims, keepdims=True)
        Sample= Sample / BatchNorm
    return Sample

def Penalty(FirstConfiguration,SecondConfiguration,Metric):
    Batchdims = np.ndim(FirstConfiguration)
    if Metric=="Square":
        Penalty = np.sum(np.linalg.norm(FirstConfiguration -SecondConfiguration,axis=Batchdims-1),axis=Batchdims-2)
    if Metric=="Entropy":
        Penalty = np.sum(np.sum(np.abs(FirstConfiguration)*(np.log(np.abs(FirstConfiguration))-np.log(np.abs(SecondConfiguration))),axis=Batchdims-1),axis=Batchdims-2)
    if Metric == "RevEntropy":
        Penalty = np.sum(np.sum(np.abs(SecondConfiguration) * (np.log(np.abs(SecondConfiguration)) - np.log(np.abs(FirstConfiguration))),axis=Batchdims-1),axis=Batchdims-2)

    return Penalty