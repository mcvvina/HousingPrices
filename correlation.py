import numpy as np
import frequencies
from scipy import stats
import normalize
import matplotlib.pyplot as plt
import removeOutlier as ro
import copy

def correlat(freq, limit, prints = False, removeOutliers = False, normalizedData= []):
    freq = freq.copy()
    if normalizedData == []:
        ### Normalize Data
        normData = normalize.normalize(freq.attributeData, freq.allPossData)
    else:
        normData = copy.deepcopy(normalizedData)
    

    if removeOutliers == True:
        #print(np.array(normData).shape)
        housesIndicesToDrop = ro.removeOutlier(list(normData),freq.attributeData[len(freq.attributeData)-1])
        housesIndicesToDrop = sorted(housesIndicesToDrop, reverse = True)
        housesIndicesToDrop = list(housesIndicesToDrop)
        for eachAttribute in normData:
            for e in housesIndicesToDrop:
                del eachAttribute[e]
        #print(np.array(normData).shape)
        plt.clf()
        
    toCorrelate =  np.array(normData)


    correlationValues = [ [] for each in freq.attributeData]
    #indicesOfSTD0 = []
    ### Iterate through normalized Data (with or without outliers)
    for ind, each in enumerate(toCorrelate):
        ### Find correlation of data to the salesPrice
        if len(set(each))!=1:
            corrs = np.corrcoef(toCorrelate[ind], toCorrelate[len(freq.attributeData)-1])            
            correlationValues[ind] = corrs[[0],[1]]
        
    #indicesOfSTD0 = sorted(indicesOfSTD0)

    

    

    ### Find most correlated
    mostCorrelated = []
    for i,each in enumerate(correlationValues):
        if each == []:
            pass
        elif abs(each) >= abs(limit): #greater than (limit) correlation
            
            mostCorrelated.append([i, each])
            if prints == True:
                print(freq.attributes[i])
    
    return mostCorrelated ### this is -> array[(index, correlationValue)]

