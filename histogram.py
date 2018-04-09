# CSC 240 Data Mining Final Project
# Spring 2018
# Jack Dalton, Mcvvina Lin
# Plots histograms of all frequencies that are nominal values
# To change figure, click on histogram figure (not the arrow buttons)




import frequencies
import matplotlib.pyplot as plt
import itertools

freq = frequencies.frequencies()#'https://raw.githubusercontent.com/mcvvina/HousingPrices/master/test.csv')

freqs = []
poss = []
attr = []
for ind,each in enumerate(freq.allPossData):
    if each != []:
        freqs.append(freq.freqData[ind])
        poss.append(freq.allPossData[ind])
        attr.append(freq.attributes[ind])

ys = itertools.cycle(freqs)
xs = itertools.cycle(poss)
ts = itertools.cycle(attr)



def onclick(event):
 
    ay = next(xs)
    plt.clf()

    plt.gcf().clear()
    
    ax.set_xticklabels(ay)
    plt.ylabel("Frequency")
    
    tit = next(ts)
    plt.title(tit+" Frequencies")
   # fig.draw()
    
    plt.bar(ay,next(ys))
    
    plt.draw()
    plt.show()
    
fig,ax = plt.subplots()

fig.canvas.mpl_connect('button_press_event', onclick)
oy = next(xs)
plt.bar(oy,next(ys))
ax.set_xticklabels(oy)
ax.set_ylabel("Frequency")
ax.set_title(next(ts)+" Frequencies")




plt.show()
plt.draw()

