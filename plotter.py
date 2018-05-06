# CSC 240 Data Mining Final Project
# Spring 2018
# Jack Dalton, Mcvvina Lin
# Plots histograms of all frequencies that are nominal values
# To change figure, click on histogram figure (not the arrow buttons)



import sys
import frequencies
import matplotlib.pyplot as plt
import itertools
import pandas
import normalize
import removeOutlier as ro
import numpy as np

def plotter(freq, fig, backend, XindicesToPlot = [], YindicesToPlot = [], typ = 'histogram',removeOutliers = False ):
    subplot = fig.add_subplot(111)
    sys.setrecursionlimit(5000)




    freqs = []
    poss = []
    attr = []
    if typ is 'plot':
        normData = normalize.normalize(freq.attributeData, freq.allPossData)
        names = pandas.Series(data = freq.attributes)
        dataOrig = pandas.DataFrame(data = normData)

        dataOrig = dataOrig.transpose()
        dataOrig.columns = (names)

        target = pandas.DataFrame(data = freq.attributeData)
        target = target.transpose()
        target.columns = names
        target = target['SalePrice']
        salesPrice = list(target)

        
        
    if XindicesToPlot == []:
        if typ is 'histogram':
            for ind,each in enumerate(freq.allPossData):
                if each != []:
                    freqs.append(freq.freqData[ind])                    
                    poss.append(freq.allPossData[ind])
                    attr.append(freq.attributes[ind])
                else:
                    ### need all the values so that each row is an attribute and each column is a value                    
                    freqs.append(freq.attributeData[ind])                    
                    poss.append([])
                    attr.append(freq.attributes[ind])
        else:
            for ind,each in enumerate(dataOrig):
                freqs.append(list(dataOrig.iloc[:,ind]))
                attr.append(dataOrig.iloc[:,ind].name)
    elif type(XindicesToPlot[0]) is not str:
        if typ is 'histogram':
            for ind in XindicesToPlot:
                if freq.allPossData[ind] != []:
                    freqs.append(freq.freqData[ind])
                    poss.append(freq.allPossData[ind])
                    attr.append(freq.attributes[ind])
                else:
                    freqs.append(freq.attributeData[ind])
                    poss.append([])
                    attr.append(freq.attributes[ind])
        else:
            for ind in XindicesToPlot:
                freqs.append(list(dataOrig.iloc[:,ind]))
                attr.append(dataOrig.iloc[:,ind].name)
    else:
        if typ is 'histogram':
            for ind in XindicesToPlot:
                if freq.allPossData[ind] != []:
                    freqs.append(freq.freqData[ind])
                    poss.append(freq.allPossData[ind])
                    attr.append(freq.attributes[ind])
                else:
                    freqs.append(freq.attributeData[ind])
                    poss.append([])
                    attr.append(freq.attributes[ind])
        else:
            ### NArrow down Attributes
            for ind in XindicesToPlot:
                if removeOutliers == True:
                    toAppendFreq = list(dataOrig.loc[:,ind])
                    freqs.append(toAppendFreq)
                else:
                    freqs.append(list(dataOrig.loc[:,ind]))

                attr.append(dataOrig.loc[:,ind].name)
            ## REmove outliers of reduced dataset

    if removeOutliers == True:
        housesIndicesToDrop = ro.removeOutlier(freqs,salesPrice)
        housesIndicesToDrop = sorted(housesIndicesToDrop, reverse = True)
        housesIndicesToDrop = list(housesIndicesToDrop)
        for e in housesIndicesToDrop:
            del salesPrice[e]
        for eachAttribute in freqs:
            for e in housesIndicesToDrop:
                del eachAttribute[e]
        freqs = list(freqs)

        
        plt.clf()

    if attr[0] == 'Id':
        del freqs[0]
        del attr[0]
        if typ is 'histogram':
            del poss[0]

        
    
    
    if typ is 'histogram':
        ys = itertools.cycle(freqs)
        xs = itertools.cycle(poss)
    elif YindicesToPlot == []:
        xs = itertools.cycle(freqs)
        
    elif YindicesToPlot != []:
        print('error')
##        tfreqs = []
##        tSales = []
##        for eachIn in YindicesToPlot:
##            tfreqs.append(freqs[eachIn])
##            tSales.append(salesPrice[eachIn])
##        xs = itertools.cycle(tfreqs)
##        salesPrice = tSales
    
    ts = itertools.cycle(attr)



    def onclick(event):
        subplot.cla()
        ay = next(xs)
        
        if ay == []:
            vals = next(ys)
            if typ is 'histogram':
                binwidth = int((max(vals) - min(vals))/20)
                if binwidth == 0:
                    binwidth = 1
                subplot.hist(vals,bins=range(int(min(vals)), int(max(vals)) + binwidth, binwidth))
            elif typ is 'plot':
                subplot.scatter(ay,salesPrice)
        else:
            if typ is 'histogram':
                subplot.bar(ay,next(ys))
                subplot.set_xticklabels(ay)
            elif typ is 'plot':
                subplot.scatter(ay,salesPrice)
            
        tit = next(ts)
        if typ is 'histogram':
            subplot.set_title(tit+" Frequencies")   
            subplot.set_ylabel("Frequency")
        elif typ is 'plot':
            subplot.set_title(tit+" vs SalePrice")   
            subplot.set_ylabel("SalePrice")
        
        
        
        fig.canvas.draw()
        fig.canvas.get_tk_widget().pack(side=backend.BOTTOM, fill=backend.BOTH, expand=True)
        fig.canvas._tkcanvas.pack(side=backend.TOP, fill=backend.BOTH, expand=True)
 
    fig.canvas.mpl_connect('button_press_event', onclick)

    oy = next(xs)
    if oy == []:
        vals = next(ys)
        if typ is 'histogram':
            binwidth = int((max(vals) - min(vals))/20)
            if binwidth == 0:
                    binwidth = 1
            subplot.hist(vals,bins=range(int(min(vals)), int(max(vals)) + binwidth, binwidth))
        elif typ is 'plot':
            print(vals)
            subplot.scatter(oy,salesPrice)
    else:
        if typ is 'histogram':
            subplot.bar(oy,next(ys))
            subplot.set_xticklabels(oy)
        elif typ is 'plot':
            subplot.scatter(oy,salesPrice)
    
    if typ is 'histogram':
        subplot.set_title(next(ts)+" Frequencies")   
        subplot.set_ylabel("Frequency")
    elif typ is 'plot':
        subplot.set_title(next(ts)+" vs SalePrice")   
        subplot.set_ylabel("SalePrice")

    fig.canvas.draw()
    fig.canvas.get_tk_widget().pack(side=backend.BOTTOM, fill=backend.BOTH, expand=True)
    fig.canvas._tkcanvas.pack(side=backend.TOP, fill=backend.BOTH, expand=True)


    

