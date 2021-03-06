import frequencies
import correlation
import histogram
import matplotlib.pyplot as plt
import normalize


### Plots the histograms of the data that is correlated to sales price
### Plots the normalized data to the sales price to visually show correlation

freq = frequencies.frequencies()
corr = correlation.correlat(freq, .6)
correlationIndices = [ each[0] for each in corr]

histogram.histograms(freq,  indicesToPlot = correlationIndices)

normData = normalize.normalize(freq.attributeData, freq.allPossData)
#plt.ion()
for ind in correlationIndices:
    plt.scatter(normData[ind],freq.attributeData[len(freq.attributeData)-1])
    plt.show()
    input("Press [enter] to continue.")
    


        
