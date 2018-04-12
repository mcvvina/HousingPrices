import numpy as np
import frequencies
from scipy import stats
import normalize
import matplotlib.pyplot as plt

def correlat(freq, limit):
    #freq = frequencies.frequencies()

    normData = normalize.normalize(freq.attributeData, freq.allPossData)

    #print(freq.attributeData[1])
    toCorrelate =  np.array(normData)

    ##for each in toCorrelate:
    ##    print(type(each[0]))
    #print(toCorrelate[0])

    correlationValues = [ [] for each in freq.attributeData]
    for ind, each in enumerate(toCorrelate):
       # print('-'*30)
       # print(str(ind) +" "+ str(freq.attributes[ind]))
        corrs = np.corrcoef(toCorrelate[ind], freq.attributeData[len(freq.attributeData)-1])    
        #print(corrs)
        correlationValues[ind] = corrs[[0],[1]]

    ##plt.scatter(toCorrelate[1], freq.attributeData[len(freq.attributeData)-1])
    ##plt.show()
    mostCorrelated = []
    for i,each in enumerate(correlationValues):
        if abs(each) >= abs(limit): #greater than (limit) correlation
            mostCorrelated.append([i, each])
            print(freq.attributes[i])
    #print(mostCorrelated)
    return mostCorrelated

